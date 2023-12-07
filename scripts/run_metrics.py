import os, sys
import numpy as np
from MindVideo import (clip_score_only, 
                       ssim_score_only, 
                       img_classify_metric, 
                       video_classify_metric,
                       remove_overlap)
import imageio.v3 as iio
import torch

import wandb

def main(
        data_path
):
    wandb.init(
        project="mind-video",
        anonymous="allow",
        save_code=True,
        group = 'eval',
        reinit = True,
    )


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    gt_list = []
    pred_list = []
    for i  in range(1200):
        gif = iio.imread(os.path.join(data_path, f'test{i+1}.gif'), index=None)
        gt, pred = np.split(gif, 2, axis=2)
        gt_list.append(gt)
        pred_list.append(pred)

    gt_list = np.stack(gt_list)
    pred_list = np.stack(pred_list)

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

if __name__ == '__main__':
    main(
        data_path = sys.argv[1]
    )