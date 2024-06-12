## Solving Masked Jigsaw Puzzles with Diffusion Vision Transformers (SPDVT) <br><sub>Official PyTorch Implementation</sub> 
[CVPR 2024] Solving Masked Jigsaw Puzzles with Diffusion Vision Transformers

### [[Paper]](https://arxiv.org/abs/2404.07292v1)

**This GitHub repository is currently undergoing organization.** Stay tuned for the upcoming release of fully functional code!

<img width="1493" alt="Main Arch" src="https://github.com/JinyangMarkLiu/JPDVT/assets/50398783/6a91130e-0940-48c7-9b7a-b842ab8fbb69">

## Setup
    git clone https://github.com/JinyangMarkLiu/JPDVT.git
    cd JPDVT

## Preparing Data
Download datasets as you need. Here we give brief instructions for setting up part of the datasets we used.

#### _ImageNet_
You can use this [script](https://gist.github.com/bonlime/4e0d236cf98cd5b15d977dfa03a63643) to download and prepare the _ImageNet_ dataset. If you need to download the dataset, please uncomment the first part of the script.

#### _JPwLEG-3_
Download the _JPwLEG-3_ from this [Google Drive](https://drive.google.com/drive/folders/1MjPm7ar-u6H5WX6Bw2qshPiYPT_eQCZE). Only [select_image](https://drive.google.com/drive/folders/1MjPm7ar-u6H5WX6Bw2qshPiYPT_eQCZE) part is used in our experiments.

## Training
We provide training scripts for training image models and video models.

### Training image models
On ImageNet dataset:

    torchrun --nnodes=1 --nproc_per_node=4 train_JPDVT.py --dataset imagenet --data-path <imagenet-train-path> --image-size 192 --crop

On MET dataset:

    torchrun --nnodes=1 --nproc_per_node=4 train_JPDVT.py --dataset met --data-path <met-data-path> --image-size 288

## Testing


## BibTeX
If you find our paper/project useful, please consider citing our paper:

```bibtex
@InProceedings{Liu_2024_CVPR,
    author    = {Liu, Jinyang and Teshome, Wondmgezahu and Ghimire, Sandesh and Sznaier, Mario and Camps, Octavia},
    title     = {Solving Masked Jigsaw Puzzles with Diffusion Vision Transformers},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {23009-23018}
}
```

## Acknowledgments
Our codebase is mainly based on [improved diffusion](https://github.com/openai/improved-diffusion), [make a video](https://github.com/lucidrains/make-a-video-pytorch), and [DiT](https://github.com/facebookresearch/DiT).
