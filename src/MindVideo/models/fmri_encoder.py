from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from diffusers.utils.import_utils import is_xformers_available
from .. import get_1d_sincos_pos_embed, interpolate_pos_embed
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import Mlp, DropPath
import torch.nn.functional as F
from typing import Dict, Optional, Tuple,Callable
import os
import json
from einops import rearrange

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

class PatchEmbed1D(nn.Module):
    """ 1 Dimensional version of data (fmri voxels) to Patch Embedding
    """
    def __init__(self, num_voxels=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = num_voxels // patch_size
        self.patch_shape = patch_size
        self.num_voxels = num_voxels
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, V = x.shape # batch, channel, voxels
        # assert V == self.num_voxels, \
        #     f"Input fmri length ({V}) doesn't match model ({self.num_voxels})."
        x = self.proj(x).transpose(1, 2).contiguous() # put embed_dim at the last dimension
        return x
    
class ProjectionHead(nn.Module):
    def __init__(self, 
                 in_channel=1, 
                 out_channel=1, 
                 in_dim=1024, 
                 out_dim=128, 
                 dropout=.0, 
                 use_norm=False,
                 use_norm_in=False):
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim, bias=False)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(out_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim) if use_norm else nn.Identity() 
        self.layer_norm_in = nn.LayerNorm(in_dim) if use_norm_in else nn.Identity() 
        self.channel_mapper = nn.Linear(in_channel, out_channel, bias=False)
        self.in_channel = in_channel
        self.out_channel = out_channel
        # print(f'ProjectionHead: in_channel={in_channel}, out_channel={out_channel}, in_dim={in_dim}, out_dim={out_dim}, dropout={dropout}, use_norm={use_norm}')
    def forward(self, x):  
        x = self.layer_norm_in(x)
        x = x.transpose(1, 2)
        x = self.channel_mapper(x) 
        x = x.transpose(1, 2)
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x # n, out_channel, out_dim


class Attention(nn.Module):
    _use_memory_efficient_attention_xformers = False
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.drop_rate = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool, 
                                                     attention_op: Optional[Callable] = None):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
        self._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
        self.attention_op = attention_op
        # print('fmri_encoder: use xformers attention')

    def batch_to_head_dim(self, tensor):
        head_size = self.num_heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor
    def head_to_batch_dim(self, tensor):
        head_size = self.num_heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor
     
    def forward(self, x, return_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        # q: B, num_heads, N, C // num_heads
        # k: B, num_heads, N, C // num_heads
        # v: B, num_heads, N, C // num_heads
        if return_attn:
            assert not self._use_memory_efficient_attention_xformers, 'return_attn is not supported with xformers'
            assert not self.training, 'return_attn is not supported in training mode'
        if self._use_memory_efficient_attention_xformers:
            q = q.reshape(B * self.num_heads, N, C // self.num_heads)
            k = k.reshape(B * self.num_heads, N, C // self.num_heads)
            v = v.reshape(B * self.num_heads, N, C // self.num_heads)
            x = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op, scale=self.scale, p=self.drop_rate
            )
            x = x.to(q.dtype)
            x = self.batch_to_head_dim(x) # B, N, C
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            if return_attn:
                return attn
            
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gradient_checkpointing = False
    
    def forward(self, x, **kwargs):
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward
            x = x + self.drop_path(torch.utils.checkpoint.checkpoint(create_custom_forward(self.attn), self.norm1(x)))
            x = x + self.drop_path(torch.utils.checkpoint.checkpoint(create_custom_forward(self.mlp), self.norm2(x)))
        else:    
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class BlockTemp(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.attn_temp = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm_temp = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gradient_checkpointing = False
    
    def forward(self, x, window_size=None, return_attn=False, **kwargs):
        if return_attn:
            assert not self.training, 'return_attn is not supported in training mode'

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward
            x = x + self.drop_path(torch.utils.checkpoint.checkpoint(create_custom_forward(self.attn), self.norm1(x)))
            # Temporal-Attention
            d = x.shape[1]
            x = rearrange(x, "(b f) d c -> (b d) f c", f=window_size)
            x = x + self.drop_path(torch.utils.checkpoint.checkpoint(create_custom_forward(self.attn_temp), self.norm_temp(x)))
            x = rearrange(x, "(b d) f c -> (b f) d c", d=d)
            x = x + self.drop_path(torch.utils.checkpoint.checkpoint(create_custom_forward(self.mlp), self.norm2(x)))
        else:    
            if return_attn:
                spatial_attn = self.attn(self.norm1(x), return_attn=return_attn)
                x = x + self.drop_path(self.attn(self.norm1(x)))
                d = x.shape[1]
                x = rearrange(x, "(b f) d c -> (b d) f c", f=window_size)
                temporal_attn = self.attn_temp(self.norm_temp(x), return_attn=return_attn)
                return spatial_attn, temporal_attn
            
            x = x + self.drop_path(self.attn(self.norm1(x)))
            # Temporal-Attention
            d = x.shape[1]
            x = rearrange(x, "(b f) d c -> (b d) f c", f=window_size)
            x = x + self.drop_path(self.attn_temp(self.norm_temp(x)))
            x = rearrange(x, "(b d) f c -> (b f) d c", d=d)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class fMRIEncoder(ModelMixin, ConfigMixin):
    config_name = 'config.json'
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self, 
                 num_voxels: int=224, 
                 patch_size: int=16, 
                 embed_dim: int=1024, 
                 in_chans: int=1,
                 depth: int=24, 
                 num_heads: int=16, 
                 mlp_ratio: float=4.,
                 # project head setting
                 out_channel: int=1,
                 out_dim: int=768,
                 use_norm: bool=False,
                 dropout: float=.0,
                 window_size: int=1,
                 use_temp_attn: bool=False,
                 logit_scale_init_value: float=1.0,
                 **kwargs):
        
        super().__init__()

        self.patch_embed = PatchEmbed1D(num_voxels, patch_size, in_chans, embed_dim)

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        norm_layer = nn.LayerNorm
        vit_block = BlockTemp if use_temp_attn else Block
        self.blocks = nn.ModuleList([
            vit_block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=dropout)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
    
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.proj_head = ProjectionHead(in_channel=num_patches * window_size, out_channel=out_channel, in_dim=embed_dim, out_dim=out_dim, use_norm=use_norm, dropout=dropout)
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)

        self.embed_dim = embed_dim
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.initialize_weights()
        self.gradient_checkpointing = False

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def forward_attn(self, x, layer):
        # embed patches
        assert self.window_size == x.shape[1], f'window_size {self.window_size} != x.shape[1] {x.shape[1]}'
        window_size = x.shape[1]
        x = rearrange(x, 'b c n -> (b c) 1 n')
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            if i != layer:
                x = blk(x, window_size=window_size)
            else:
                spatial_attn, temp_attn = blk(x, window_size=window_size, return_attn=True)
                break
        # spatial_attn = rearrange(spatial_attn, '(b c) n d -> b c n d', c=window_size)
        return (spatial_attn / spatial_attn.norm(p=2, dim=-1, keepdim=True), 
                temp_attn / temp_attn.norm(p=2, dim=-1, keepdim=True))
        

    def forward_encoder(self, x):
        # embed patches
        assert self.window_size == x.shape[1], f'window_size {self.window_size} != x.shape[1] {x.shape[1]}'
        window_size = x.shape[1]
        x = rearrange(x, 'b c n -> (b c) 1 n')
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, window_size=window_size)
        x = self.norm(x) # output: N * window_size, n_seq, embed_dim
        x = rearrange(x, '(b c) n d -> b (c n) d', c=window_size)

        if self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.proj_head), x)
        else:
            x = self.proj_head(x)
        return x  

    def forward(self, imgs):
        if imgs.ndim == 1:
            imgs = imgs[None, None, :]
        if imgs.ndim == 2:
            imgs = imgs[:, None, :]
        # expected input shape: N, 1, num_voxels
        latent = self.forward_encoder(imgs) # N, n_seq, embed_dim
        return latent # N, n_seq, embed_dim
    
    def load_checkpoint(self, state_dict):
        state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k)}
        interpolate_pos_embed(self, state_dict)
            
        m, u = self.load_state_dict(state_dict, strict=False)
        print('missing keys:', u)
        print('unexpected keys:', m)
        return 
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path, num_voxels, subfolder=None):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        model = cls.from_config_path(pretrained_model_path, num_voxels)
        ckp_file = os.path.join(pretrained_model_path, 'model.pth' 
                                if 'model.pth' in os.listdir(pretrained_model_path) else 'diffusion_pytorch_model.bin')
        if not os.path.isfile(ckp_file):
            raise RuntimeError(f"{ckp_file} does not exist")
        ckp = torch.load(ckp_file, map_location='cpu')
        ckp = ckp['model'] if 'model' in ckp.keys() else ckp
        model.load_checkpoint(ckp)
        return model
    
    @classmethod
    def from_config_path(cls, config_path, num_voxels, subfolder=None):
        if subfolder is not None:
            config_path = os.path.join(config_path, subfolder)
        config_file = os.path.join(config_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)
        config['num_voxels'] = num_voxels
        model = cls.from_config(config)
        return model
    
    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True
        if isinstance(module, Block):
            # print('fmri_encoder: enable gradient_checkpointing')
            module.gradient_checkpointing = value