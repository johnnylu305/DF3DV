# DF3DV-1K

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
- [DF3DV-1K Hugging Face](https://huggingface.co/datasets/ChengYou305/DF3DV-1K)
- [DF3DV-1K Google Cloud](https://drive.google.com/drive/folders/1OuLBTius8bnXfd51ckgRo7H32ZRv2q8N?usp=sharing)
  
The original dataset can be downloaded from both Hugging Face and Google Cloud after completing the form and agreeing to the terms of use [here]().


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
- [DF3DV-1K Hugging Face](https://huggingface.co/datasets/ChengYou305/DF3DV-1K-Fixer)
- [DF3DV-1K Google Cloud](https://drive.google.com/drive/folders/1JE399E_o45mLavMqEWzpEYAlB-4-rjkc?usp=sharing)
  
The original dataset can be downloaded from both Hugging Face and Google Cloud after completing the form and agreeing to the terms of use [here]().
