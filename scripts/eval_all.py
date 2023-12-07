import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from MindVideo import create_Wen_dataset, create_Wen_test_data_only
from MindVideo import fMRIEncoder
import argparse
import datetime
import wandb
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image

from omegaconf import OmegaConf
from typing import Dict, Optional, Tuple
from accelerate.utils import set_seed
import inspect
from MindVideo import UNet3DConditionModel
from MindVideo import MindVideoPipeline
from tqdm.auto import tqdm
from MindVideo import save_videos_grid
from diffusers import AutoencoderKL

from MindVideo import (clip_score_only, 
                       ssim_score_only, 
                       img_classify_metric, 
                       video_classify_metric)

def to_image(img):
    if img.shape[-1] != 3:
        img = rearrange(img, 'c h w -> h w c')
    img = 255. * img
    return Image.fromarray(img.astype(np.uint8))

def channel_last(img):
    if img.shape[-1] == 3:
        return img
    if len(img.shape) == 3:
        img = rearrange(img, 'c h w -> h w c')
    elif len(img.shape) == 4:
        img = rearrange(img, 'f c h w -> f h w c')
    else:
        raise ValueError(f'img shape should be 3 or 4, but got {len(img.shape)}')
    return img

def channel_first(img):
    if len(img.shape) == 3:
        if img.shape[0] == 3:
            return img
        img = rearrange(img, 'h w c -> c h w')
    elif len(img.shape) == 4:
        if img.shape[1] == 3:
            return img
        img = rearrange(img, 'f h w c -> f c h w')
    else:
        raise ValueError(f'img shape should be 3 or 4, but got {len(img.shape)}')
    return img


def normalize(img):
    if img.shape[-1] == 3 and len(img.shape) == 3:
        img = rearrange(img, 'h w c -> c h w')
    elif img.shape[-1] == 3 and len(img.shape) == 4:
        img = rearrange(img, 'f h w c -> f c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img
    
def main(  
        val_data_setting: Dict,
        checkpoint_path: str='checkpoints/19-03-2023-21:23:30',
        data_dir: str='./data',
        seed: int=2023,
        dataset: str='Wen',
        patch_size: int=16,
        subjects: list=['subject1'],
        working_dir: str='.',
        eval_batch_size: int = 4,
        output_path: Optional[str] = None,
        group_name: str = 'default',
        window_size: int = 1,
        load_test_data_only: bool = True, 
        half_precision: bool = False,
        **kwargs
):
    # project setup
    *_, config = inspect.getargvalues(inspect.currentframe())
    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(seed)


    output_path = os.path.join(working_dir, 'results', 'eval', '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S"))) if output_path is None else output_path
    os.makedirs(output_path, exist_ok=True)
    wandb.init(
        project="mind-video",
        anonymous="allow",
        save_code=True,
        config=config,
        group = group_name,
        reinit = True,
        notes = 'this runs video reconstruction on the test set',
    )

    OmegaConf.save(config, os.path.join(output_path, 'config.yaml'))
    
    w = val_data_setting['width']
    h = val_data_setting['height']
    img_transform_test = transforms.Compose([
        normalize, transforms.Resize((w, h)), 
        channel_first
    ])
    fps = val_data_setting.video_length // 2

    if dataset == 'Wen':
        if load_test_data_only:
            dataset_test = create_Wen_test_data_only(data_dir, patch_size, 
                    fmri_transform=torch.FloatTensor, image_transform=[img_transform_test, img_transform_test], 
                    subjects=subjects, window_size=window_size, fps=fps)
        else:
            _, dataset_test = create_Wen_dataset(data_dir, patch_size, 
                    fmri_transform=torch.FloatTensor, image_transform=[img_transform_test, img_transform_test], 
                    subjects=subjects, window_size=window_size, fps=fps)
            
        num_voxels = dataset_test.num_voxels
    else:
        raise NotImplementedError(f'{dataset} not implemented')

    dtype = torch.float16 if half_precision else torch.float32
    unet = UNet3DConditionModel.from_pretrained_2d(checkpoint_path, subfolder="unet").to(device, dtype=dtype)
    fmri_encoder = fMRIEncoder.from_pretrained(checkpoint_path, subfolder='fmri_encoder', num_voxels=num_voxels).to(device, dtype=dtype)
    
    vae = AutoencoderKL.from_pretrained(checkpoint_path, subfolder="vae").to(device, dtype=dtype)

    fmri_encoder.eval()
    vae.eval()
    unet.eval()
    # Get the validation pipeline
    pipe = MindVideoPipeline.from_pretrained(checkpoint_path, 
                                            unet=unet, fmri_encoder=fmri_encoder, vae=vae, torch_dtype=dtype).to(device)
    # pipe.enable_vae_slicing()
    pipe.enable_xformers_memory_efficient_attention()

    # DataLoaders creation:
    eval_dataloader = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size, shuffle=False
    )

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(len(eval_dataloader)))
    progress_bar.set_description("Steps")

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    gt = []
    pred = []
    for idx, prompt in enumerate(eval_dataloader):
        video = prompt['image']
        # _ = save_videos_grid(rearrange(video, 'b t c h w -> b c t h w'), f"{output_path}/samples/sample-all/test{idx+1}-gt.gif", 
        #                     rescale=True, fps=3)
        video = (rearrange(video, 'b t c h w -> b c t h w') + 1.0) / 2.0
        sample = pipe(prompt['fmri'], negative_prompt=prompt['uncon_fmri'] ,generator=generator, 
                                        **val_data_setting).videos
        out = save_videos_grid(torch.concat([video, sample]), f"{output_path}/samples/sample-all/test{idx+1}.gif", fps=fps)
        # out = save_videos_grid(sample, f"{output_path}/samples/sample-all/test{idx+1}.gif", fps=val_data_setting.video_length // 2)
        out = rearrange(np.stack(out), 't h w c -> t c h w') / 255.
        out = F.interpolate(torch.from_numpy(out), size=(128, 128 * 2), mode='bilinear', align_corners=False)
        wandb.log({
            f"all-test{idx+1}": wandb.Video((out * 255).numpy().astype(np.uint8), fps=fps, format="gif") 
        })
        gt.append((rearrange(video, 'b c t h w -> (b t) c h w') * 255).numpy().astype(np.uint8))
        pred.append((rearrange(sample, 'b c t h w -> (b t) c h w') * 255).numpy().astype(np.uint8))
        progress_bar.update(1)

    gt_list = np.stack(gt)
    pred_list = np.stack(pred)

    print(f'pred shape: {pred_list.shape}, gt shape: {gt_list.shape}')

    # image classification scores
    n_way = 50
    num_trials = 100
    top_k = 1
    # video classification scores
    acc_list, std_list = video_classify_metric(
                                        pred_list,
                                        gt_list,
                                        n_way = n_way,
                                        top_k=top_k,
                                        num_trials=num_trials,
                                        num_frames=gt_list.shape[1],
                                        return_std=True,
                                        device=device
                                        )
    print(f'video classification score: {np.mean(acc_list)} +- {np.mean(std_list)}')
    wandb.log({'video_classification_score': np.mean(acc_list)})

    for i in range(pred_list.shape[1]):

        # ssim scores
        ssim_scores, std = ssim_score_only(pred_list[:, i], gt_list[:, i])
        print(f'ssim score: {ssim_scores}, std: {std}')
        wandb.log({'ssim_score': ssim_scores, 'ssim_std': std})
        
        acc_list, std_list = img_classify_metric(
                                            pred_list[:, i], 
                                            gt_list[:, i], 
                                            n_way = n_way, 
                                            top_k=top_k, 
                                            num_trials=num_trials, 
                                            return_std=True,
                                            device=device)
        print(f'img classification score: {np.mean(acc_list)} +- {np.mean(std_list)}')
        wandb.log({'img_classification_score': np.mean(acc_list)})



def get_args_parser():
    parser = argparse.ArgumentParser('Decoding fMRI to reconstruct videos')
    # project parameters
    parser.add_argument('--config', type=str, default='configs/eval_all.yaml', help='path to config file')
    return parser

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    config = OmegaConf.load(args.config)
    config.config_path = args.config

    main(**config)
    
        




    