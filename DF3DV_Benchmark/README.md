## Download DF3DV-1K
Follow the download instructions [here](https://github.com/johnnylu305/DF3DV-1K/tree/main) to download DF3DV-1K.

## Downscale Images

Downscale images by a factor of 8, which is a commonly used downsampling factor for distractor-free radiance fields.

```
# Create undistortion_images_8 from undistortion_images
python downsample_images.py --root DF3DV-1K --factor 8 --num_workers 8 --overwrite
```

The current directory structure is ready for benchmarking.
```
DF3DV-1K
├── DF3DV-1K-Star
│   ├── 0000
│   │   ├── 040625-LundoBin
│   │   │   └── 040625-LundoBin-All
│       │       ├── split.json
│   │   │       ├── undistortion_images_8
│   │   │       │   ├── clutter_IMG_7042.JPG
│   │   │       │   ├── ...
│   │   │       │   └── extra_IMG_7041.JPG
│   │   │       └── undistortion_sparse
│   │   ├── ...
│   │   └── 090625-BlueBikeBell
│   ├── ...
│   └── 0024
└── DF3DV-41
    ├── 021125-Chess
    │   └── 021125-Chess-All
    │       ├── split.json
    │       ├── undistortion_images_8
    │       │   ├── clutter_IMG_7525.JPG
    │       │   ├── ...
    │       │   └── extra_IMG_7524.JPG
    │       └── undistortion_sparse
    ├── ...
    └── 301025-TempleDrumIncense
```

## Training Instructions

- Use clutter images (`clutter_*.JPG`) in `undistortion_images_8` to train a distractor-free radiance field.
- Use clean images (`extra_*.JPG`) in `undistortion_images_8` to benchmark novel-view synthesis quality.
- Use `split.json` to further divide the data into training and testing sets.
- Use `undistortion_sparse` as the camera parameters, and carefully account for the image downscaling factor when adjusting the camera intrinsics, if necessary.
- Use the high-resolution images in `undistortion_images` only when necessary.
- For the DF3DV-1K benchmark, use all scenes in `DF3DV-1K-Star` and `DF3DV-41`.
- For the DF3DV-41 benchmark, use all scenes in `DF3DV-41`.

## Output Instructions

The following is the recommended output directory structure.

- Place your `<method>` in the `MODELS` folder.
- Place the checkpoint used for evaluation in `<method>/ckpts/`.
- Place the composition images (`|GT|Mask|Rendering|`) of the training views in `renders/composition_<step>/`.
- Place the ground-truth training-view images (`|GT|`) in `renders/data_<step>/`.
- Place the mask images (`|Mask|`) of the training views in `renders/mask_<step>/`.
- Place the rendered training-view images (`|Rendering|`) in `renders/render_<step>/`.
- Place the evaluated images (`|GT|Rendering|`) of the evaluation views in `renders/`.
- Place `val_<step>.json` containing `{"psnr": xx, "ssim": xx, "lpips": xx}` of evaluated images in the `stats` folder.
- Place any additional files in the `MODELS` folder if necessary.

```
DF3DV-1K
├── DF3DV-1K-Star
│   ├── 0000
│   │   ├── 040625-LundoBin
│   │   │   └── 040625-LundoBin-All
│   │   │       └── MODELS
│   │   │           ├── 3DGS
│   │   │           │   ├── 3dgs_log.txt
│   │   │           │   ├── cfg.json
│   │   │           │   ├── ckpts
│   │   │           │   │   └── ckpt_29999.pt
│   │   │           │   ├── renders
│   │   │           │   │   ├── composition_29999
│   │   │           │   │   │   ├── clutter_IMG_7042.png
│   │   │           │   │   │   ├── ...
│   │   │           │   │   │   └── clutter_IMG_7140.png
│   │   │           │   │   ├── data_29999
│   │   │           │   │   │   ├── clutter_IMG_7042.png
│   │   │           │   │   │   ├── ...
│   │   │           │   │   │   └── clutter_IMG_7140.png
│   │   │           │   │   ├── mask_29999
│   │   │           │   │   │   ├── clutter_IMG_7042.png
│   │   │           │   │   │   ├── ...
│   │   │           │   │   │   └── clutter_IMG_7140.png
│   │   │           │   │   ├── render_29999
│   │   │           │   │   │   ├── clutter_IMG_7042.png
│   │   │           │   │   │   ├── ...
│   │   │           │   │   │   └── clutter_IMG_7140.png
│   │   │           │   │   ├── extra_IMG_6957.png
│   │   │           │   │   ├── ...
│   │   │           │   │   └── extra_IMG_7041.png
│   │   │           │   ├── stats
│   │   │           │   │   ├── train_step29999.json
│   │   │           │   │   └── val_step29999.json
│   │   │           │   ├── tb
│   │   │           │   │   └── events.out.tfevents.1781183745.cibci9.ihpc.uts.edu.au.1105142.0
│   │   │           │   └── videos
│   │   │           │       └── traj_29999.gif
│   │   │           ├── ...
│   │   │           └── WILDGS
│   │   ├── ...
│   │   └── 090625-BlueBikeBell
│   ├── ...
│   └── 0024
└── DF3DV-41
```
You can preview a sample scene [here](https://drive.google.com/file/d/1ebdHQUJCqMCU3VyPDSE3GqoFoJMpfIWC/view?usp=sharing).

## Benchmarked 3DGS/DF-3DGS codebases
- [AsymGS](https://github.com/johnnylu305/AsymGS-DF3DV)
- [DeGauss](https://github.com/johnnylu305/DeGauss-DF3DV)
- [DeSplat](https://github.com/johnnylu305/desplat-DF3DV)
- [OCSplats](https://github.com/johnnylu305/OCSplats-DF3DV)
- [RobustSplat](https://github.com/johnnylu305/RobustSplat-DF3DV)
- [SpotLessSplats](https://github.com/johnnylu305/SpotLessSplats-DF3DV)
- [T-3DGS](https://github.com/johnnylu305/T-3DGS-DF3DV)
- [T-3DGS-TMR](https://github.com/johnnylu305/T-3DGS-DF3DV)
- [WildGaussians](https://github.com/johnnylu305/wild-gaussians-DF3DV)
- [3DGS](https://github.com/johnnylu305/SpotLessSplats-DF3DV)

## Evaluation
```
# evaluate metrics between undistortion_images_8/extra*.JPG and renders/extra*.png
python benchmark_df3dv.py --root DF3DV-1K --eval_star --eval_41 --start 0 --end 24 --num_workers 8
```

## Leaderboard File Preparation

- You can extract the rendering images of `<method>` into the `leaderboard/<method>` folder using the following commands. 
```
# DF3DV-1K
python extract_leaderboard_images.py --root DF3DV-1K --method <method> --eval_star --eval_41 --start 0 --end 24

# DF3DV-41
python extract_leaderboard_images.py --root DF3DV-1K --method <method> --eval_41
```
- After extraction, please compress the folder into a ZIP file.
- We will use the provided evaluation script to recompute PSNR, LPIPS (AlexNet), and SSIM between the rendered images in `leaderboard/<method>` and their corresponding ground-truth images in `undistortion_images_8`.
```
# DF3DV-1K
python benchmark_leaderboard.py --root DF3DV-1K --method <method> --eval_star --eval_41 --start 0 --end 24 --num_workers 8

# DF3DV-41
python benchmark_leaderboard.py --root DF3DV-1K --method <method> --eval_41 --num_workers 8
```

## Leaderboard File Submission

- The official leaderboard currently accepts DF3DV-41 submissions only.
- Please complete the submission [form](https://docs.google.com/forms/d/e/1FAIpQLSfy5-jwm5eGIz8a9mGIdxP49GmilLoaQKHtl31Y6G3doyz1qQ/viewform?usp=publish-editor) first.
- Then, zip the submission files and send them to [Cheng-You Lu](mailto:a2694815@gmail.com) with the subject line: "[Your Method Name] DF3DV-41 Leaderboard Submission". Please also include a Google Drive download link in your email.
