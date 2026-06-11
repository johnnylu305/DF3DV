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
```

```
