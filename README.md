---
<div align="center">

# DF3DV-1K: A Large-Scale Dataset and Benchmark for Distractor-Free Novel View Synthesis

[Cheng-You Lu](https://johnnylu305.github.io/)<sup>1</sup>, Yi-Shan Hung<sup>2</sup>, Wei-Ling Chi<sup>3*</sup>, Hao-Ping Wang<sup>3*</sup>,  Charlie Li-Ting Tsai<sup>1*</sup>, Yu-Cheng Chang<sup>1</sup>, [Yu-Lun Liu](https://yulunalexliu.github.io/)<sup>3</sup>, Thomas Do<sup>1</sup>, [Chin-Teng Lin](https://scholar.google.com/citations?user=nubkF1cAAAAJ&hl=zh-TW)<sup>1</sup>

<p><sup>1</sup>University of Technology Sydney &nbsp;&nbsp;<sup>2</sup>University of Sydney&nbsp;&nbsp;<sup>3</sup>National Yang Ming Chiao Tung University&nbsp;&nbsp;
<br><sup>*</sup>Equal Contribution&nbsp;&nbsp;

### [Project Page](https://johnnylu305.github.io/df3dv1k_web/) · [Paper](https://arxiv.org/abs/2604.13416) · [DF3DV-1K Dataset](https://github.com/johnnylu305/DF3DV#download) · [DF3DV Benchmark](https://johnnylu305.github.io/df3dv1k_web/#leaderboard)
### [DI²FIX Dataset](https://github.com/johnnylu305/DF3DV/tree/main#download-1) · [DI²FIX Demo Page](https://chengyou305-di2fix-demo.hf.space/) · [DI²FIX Codebase](https://github.com/johnnylu305/DF3DV/tree/main/DI2FIX) · [DI²FIX Checkpoint](https://huggingface.co/ChengYou305/DI2FIX_HF/tree/main)

---
</div>

## ✨ News
- **DF3DV-Extra is now available**, adding more than **1,000 new scenes** and doubling the size of the DF3DV dataset! Please temporarily assign the credit for the DF3DV-Extra dataset to the DF3DV-1K team by mentioning that your work uses DF3DV-Extra, collected by the DF3DV-1K group, if you find the dataset useful.

## Outline

- Dataset
    - [DF3DV-1K Dataset](https://github.com/johnnylu305/DF3DV#df3dv-1k)
    - [DI²FIX Dataset](https://github.com/johnnylu305/DF3DV/tree/main#download-1)
- Quick Start
    - [DF3DV-1K Quick Start](https://github.com/johnnylu305/DF3DV/tree/main/DF3DV_Benchmark)
    - [DI²FIX Quick Start](https://github.com/johnnylu305/DF3DV/tree/main/DI2FIX)
- Benchmark
    - [Leaderboard](https://johnnylu305.github.io/df3dv1k_web/#leaderboard)
    - [Leaderboard Submission](https://github.com/johnnylu305/DF3DV/tree/main/DF3DV_Benchmark#leaderboard-file-preparation) 
- Data Customization
  - [DF3DV-1K](https://github.com/johnnylu305/DF3DV/tree/main/DF3DV_Data_Preparation)
  - [DI²FIX](https://github.com/johnnylu305/DF3DV/tree/main/Fixer_Data_Preparation)

## DF3DV-1K

We release one thousand scenes with clean and cluttered images for distractor-free 3D vision research. The first few chunks were captured using interval shooting, resulting in a more casual and lower-quality capture setting (e.g., 0000 vs. 0024).

### Directory Structure
<details>
<summary>Example Directory Structure</summary>

```
├── DF3DV-1K-Star
│   ├── 0000
│   │   ├── 040625-LundoBin
│   │   │   ├── 040625-LundoBin-All (curated data)
│   │   │   │   ├── images (COLMAP input images)
│   │   │   │   │   ├── clutter_IMG_7042.JPG 
│   │   │   │   │   ├── ...
│   │   │   │   │   └── extra_IMG_7041.JPG
│   │   │   │   ├── sparse (COLMAP result)
│   │   │   │   │   └── 0
│   │   │   │   │       ├── cameras.bin
│   │   │   │   │       ├── images.bin
│   │   │   │   │       ├── points3D.bin
│   │   │   │   │       └── project.ini
│   │   │   │   ├── split.json (list of clean and cluttered images)
│   │   │   │   ├── transforms_clutter.json (Instant-NGP JSON file for cluttered images only)
│   │   │   │   ├── transforms_extra.json (Instant-NGP JSON file for clean images only)
│   │   │   │   ├── transforms.json (Instant-NGP JSON file for all images)
│   │   │   │   ├── undistortion_images (COLMAP-undistorted images)
│   │   │   │   │   ├── clutter_IMG_7042.JPG
│   │   │   │   │   ├── ...
│   │   │   │   │   └── extra_IMG_7041.JPG
│   │   │   │   └── undistortion_sparse (COLMAP-undistorted result)
│   │   │   │       └── 0
│   │   │   │           ├── cameras.bin
│   │   │   │           ├── cameras.txt
│   │   │   │           ├── images.bin
│   │   │   │           ├── images.txt
│   │   │   │           ├── points3D.bin
│   │   │   │           └── points3D.txt
│   │   │   ├── 040625-LundoBin-Clean (candidate clean images)
│   │   │   │   └── images
│   │   │   │       ├── IMG_6957.JPG
│   │   │   │       ├── ...
│   │   │   │       └── IMG_7041.JPG
│   │   │   └── 040625-LundoBin-Clutter (candidate cluttered images)
│   │   │       └── images
│   │   │            ├── IMG_7042.JPG
│   │   │            ├── ...
│   │   │            └── IMG_7140.JPG
│   │   ├── ...
│   │   └── 090625-BlueBikeBell
│   ├── ...
│   └── 0024
└── DF3DV-41
    ├── 021125-Chess
    │   ├── 021125-Chess-All
    │   │   ├── images
    │   │   ├── sparse
    │   │   │   └── 0
    │   │   ├── undistortion_images
    │   │   └── undistortion_sparse
    │   │       └── 0
    │   ├── 021125-Chess-Clean
    │   │   └── images
    │   └── 021125-Chess-Clutter
    │       └── images
    ├── ...
    └── 301025-TempleDrumIncense
```
</details>

### Download
The masked dataset can be downloaded from both Hugging Face and Google Cloud.
- [DF3DV-1K Hugging Face](https://huggingface.co/datasets/ChengYou305/DF3DV-1K) (~1TB)
- [DF3DV-1K Google Cloud](https://drive.google.com/drive/folders/1OuLBTius8bnXfd51ckgRo7H32ZRv2q8N?usp=sharing) (~1TB)
- [DF3DV-Extra Hugging Face](https://huggingface.co/datasets/ChengYou305/DF3DV-Extra)
- [DF3DV-Extra Google Cloud](https://drive.google.com/drive/folders/1FroXyRzqxSmlRtrQOAUZlN5AbYBJ9f-W?usp=sharing)
  
The original dataset can be downloaded from both Hugging Face and Google Cloud after completing the form and agreeing to the terms of use [here](https://docs.google.com/forms/d/e/1FAIpQLSfQDGWlcIUNees5SqmOmgnQ4vlk7VnkA703ybgHbJLV-MSoZg/viewform?usp=dialog).


## DF3DV-1K-Fixer

We also release paired 3DGS/Distractor-Free 3DGS renderings and corresponding ground-truth images for distractor-removal and fixer research.

### Directory Structure
<details>
<summary>Example Directory Structure</summary>

```
├── Train
│   └── DF3DV-1K-Star-Fixer
│       ├── 0000
│       │   ├── 040625-LundoBin
│       │   │   └── 040625-LundoBin-All
│       │   │       ├── MODELS
│       │   │       │   ├── 3DGS (rendering results and ground-truth images)
│       │   │       │   │   ├── 040625-LundoBin_3DGS_extra_IMG_6957_source.png
│       │   │       │   │   ├── ...
|       |   │       │   │   └── 040625-LundoBin_3DGS_extra_IMG_7041_target.png
│       │   │       │   ├── ...
│       │   │       │   └── WILDGS
│       │   │       ├── NN (nearest neighbors for images)
|       |   |       │   └── nn_rank_by_name.json
│       │   │       └── undistortion_sparse (COLMAP result)
│       │   │           └── 0
│       │   │               ├── cameras.bin
│       │   │               ├── cameras.txt
│       │   │               ├── images.bin
│       │   │               ├── images.txt
│       │   │               ├── points3D.bin
│       │   │               ├── points3D.ply
│       │   │               └── points3D.txt
│       │   ├── ...
│       │   └── 090625-BlueBikeBell
│       ├── ...
│       └── 0024
└── Val
    ├── DF3DV-41-Fixer
    │   ├── 021125-Chess
    │   │   └── 021125-Chess-All
    │   │       ├── MODELS
    │   │       │   ├── 3DGS
    │   │       │   ├── ...
    │   │       │   └── WILDGS
    │   │       ├── NN
    │   │       └── undistortion_sparse
    │   │           └── 0
    │   ├── ...   
    │   └── 301025-TempleDrumIncense
    └── on-the-go
        ├── corner
        │   └── corner-All
        │       ├── MODELS
        │       │   ├── 3DGS
        │       │   ├── ...
        │       │   └── WILDGS
        │       ├── NN
        │       └── undistortion_sparse
        │           └── 0
        ├── ...
        └── spot
```
</details>

### Download
The masked dataset can be downloaded from both Hugging Face and Google Cloud.
- [DF3DV-1K Hugging Face](https://huggingface.co/datasets/ChengYou305/DF3DV-1K-Fixer) (~0.2TB)
- [DF3DV-1K Google Cloud](https://drive.google.com/drive/folders/1JE399E_o45mLavMqEWzpEYAlB-4-rjkc?usp=sharing) (~0.2TB)
  
The original dataset can be downloaded from both Hugging Face and Google Cloud after completing the form and agreeing to the terms of use [here](https://docs.google.com/forms/d/e/1FAIpQLSfQDGWlcIUNees5SqmOmgnQ4vlk7VnkA703ybgHbJLV-MSoZg/viewform?usp=dialog).

## Citation
```
@article{lu2026df3dv,
  title={DF3DV-1K: A Large-Scale Dataset and Benchmark for Distractor-Free Novel View Synthesis},
  author={Lu, Cheng-You and Hung, Yi-Shan and Chi, Wei-Ling and Wang, Hao-Ping and Tsai, Charlie Li-Ting and Chang, Yu-Cheng and Liu, Yu-Lun and Do, Thomas and Lin, Chin-Teng},
  journal={arXiv preprint arXiv:2604.13416},
  year={2026}
}
```
