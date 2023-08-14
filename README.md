# Cinematic Mindscapes: High-quality Video Reconstruction from Brain Activity 
[arXiv](https://arxiv.org/abs/2305.11675) | [Website](https://mind-video.com/).<br/>
<p align="center">
<img src=assets/first_fig.png />
</p>

## MinD-Video
**MinD-Video** is a framework for high-quality video reconstruction from brain recording. <br/>

[**Cinematic Mindscapes: High-quality Video Reconstruction from Brain Activity**](https://arxiv.org/abs/2305.11675).<br/>
[Zijiao Chen](https://scholar.google.com/citations?user=gCTUx9oAAAAJ&hl=en)\*,
[Jiaxin Qing](https://scholar.google.com/citations?user=jpUlRiYAAAAJ&hl=en)\*,
[Juan Helen Zhou](https://scholar.google.com.sg/citations?user=4Z1S3_oAAAAJ&hl=en)<br/>
\* equal contribution <br/>

## News
- May. 20, 2023. Preprint release.

## Abstract
Reconstructing human vision from brain activities has been an appealing task that helps to understand our cognitive process. Even though recent research has seen great success in reconstructing static images from non-invasive brain recordings, work on recovering continuous visual experiences in the form of videos is limited.
In this work, we propose MinD-Video that learns spatiotemporal information from continuous fMRI data of the cerebral cortex
progressively through masked brain modeling, multimodal contrastive learning with spatiotemporal attention, and co-training with an augmented Stable Diffusion model that incorporates network temporal inflation. 
We show that high-quality videos of arbitrary frame rates can be reconstructed with MinD-Video using adversarial guidance. The recovered videos were evaluated with various semantic and pixel-level metrics. We achieved an average accuracy of 85% in semantic classification tasks and 0.19 in structural similarity index (SSIM), outperforming the previous state-of-the-art by 45%. We also show that our model is biologically plausible and interpretable, reflecting established physiological processes.

## Overview

![flowchar-img](assets/flowchart.jpg) 


## Samples
- Some samples are shown below. Our methods can reconstruct various objects, animals, motions, and scenes. The reconstructed videos are of high quality and are consistent with the ground truth. For more samples, please refer to our [website](https://mind-video.com/) or download with [google drive](https://drive.google.com/drive/folders/1d7LUkHOMCLUtxvYbgeGAFGIJ4UAmCM0w?usp=sharing).
- The following samples are currently generated with one RTX3090. Due to GPU memory limitation, samples shown below are currently 2 seconds of 3 FPS at the resolution of 256 x 256. But our method can work with longer brain recordings and reconstruct longer videos with full frame rate (30 FPS)  and higher resolution, if more GPU memory is available.
<table>
  <tr>
      <td> &nbsp; &nbsp; &nbsp; &nbsp; GT&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Ours</td>
      <td> &nbsp; &nbsp; &nbsp; &nbsp; GT&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Ours</td>
      <td> &nbsp; &nbsp; &nbsp; &nbsp; GT&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Ours</td>
      <td> &nbsp; &nbsp; &nbsp; &nbsp; GT&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Ours</td>
      <td> &nbsp; &nbsp; &nbsp; &nbsp; GT&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Ours</td>
  </tr>
  <tr>
      <td> <img src="assets/gif/test140.gif" width = 200 height = 100 ></td>
      <td> <img src="assets/gif/test227.gif" width = 200 height = 100 ></td>
      <td> <img src="assets/gif/test271.gif" width = 200 height = 100 ></td>
      <td> <img src="assets/gif/test368.gif" width = 200 height = 100 ></td>
      <td> <img src="assets/gif/test333.gif" width = 200 height = 100 ></td>
  </tr> 
  <tr>
      <td> <img src="assets/gif/test381.gif" width = 200 height = 100 ></td>
      <td> <img src="assets/gif/test385.gif" width = 200 height = 100 ></td>
      <td> <img src="assets/gif/test403.gif" width = 200 height = 100 ></td>
      <td> <img src="assets/gif/test406.gif" width = 200 height = 100 ></td>
      <td> <img src="assets/gif/test463.gif" width = 200 height = 100 ></td>
    
  </tr>

  <tr>
      <td> <img src="assets/gif/test556.gif" width = 200 height = 100 ></td>
      <td> <img src="assets/gif/test669.gif" width = 200 height = 100 ></td>
      <td> <img src="assets/gif/test708.gif" width = 200 height = 100 ></td>
      <td> <img src="assets/gif/test1011.gif" width = 200 height = 100 ></td>
      <td> <img src="assets/gif/test582.gif" width = 200 height = 100 ></td>
    
  </tr>
</table>

## Environment setup
To be updated

## Download data and checkpoints
To be updated


## Comments
- Codes will be released soon.

## BibTeX
```
@article{chen2023cinematic,
  title={Cinematic Mindscapes: High-quality Video Reconstruction from Brain Activity},
  author={Chen, Zijiao and Qing, Jiaxin and Zhou, Juan Helen},
  journal={arXiv preprint arXiv:2305.11675},
  year={2023}
}
```
