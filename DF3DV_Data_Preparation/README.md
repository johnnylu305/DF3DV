## Data Capture
<details>
<summary>Capture Instruction</summary>
  
- Use consumer devices such as smartphones and capture images in JPG format at 4K resolution.
- Disable post-processing features that may alter image geometry or resolution (e.g., image stabilization/anti-shake). All images within the same scene should have the same image size.
- Capture the clean scene first and the clutter scene afterward.
- Ensure that the clean scene remains unchanged during capture, and that distractors are not part of the clean scene.
- For most scenes, 10-15 clean images and 10-15 clutter images are sufficient.
- Keep the camera trajectory and scene coverage similar between clean and clutter captures.
- Do not zoom in or out during capture.
- Avoid blurry or out-of-focus images. Background blur may occur when the camera is too close to nearby objects or the floor; if this happens, choose a better viewpoint. 
- Manual shooting is preferred over interval shooting, which often produces motion blur.
- Clouds and shadows are generally acceptable (except for shadow-specific benchmarks), but avoid capturing during strong winds.
- Distractor objects themselves may be of low quality (e.g., motion blur or partial occlusion), as such effects naturally occur in real-world environments.
- We recommend giving a gesture in the first clutter image to help reviewers quickly identify the clutter sequence.
- After capture, you should have a set of clean and clutter JPG images ready for further processing.

</details>

You can preview sample captured scenes [here](https://drive.google.com/drive/folders/1Sx9YrZ6WZZkvc-w6om59yciTe9EkuroB?usp=sharing).

## Data Review and Organization

Download the data directory structure template [here](https://drive.google.com/drive/folders/1oVPpGzphZmO_mHRcRQxuJq4lZID5_z4c?usp=sharing).

<details>
<summary>Directory Structure</summary>
  
```
mmddyy-SceneName/
├── mmddyy-SceneName-Clean
│   └── images
└── mmddyy-SceneName-Clutter
    └── images
```

</details>

<details>
<summary>Organization Instructions</summary>

- Place clean JPG images in `mmddyy-SceneName-Clean/images` and clutter JPG images in `mmddyy-SceneName-Clutter/images`.
- Remove low-quality images and move a few clean images to the clutter folder if necessary.
- Rename `mmddyy-SceneName`, `mmddyy-SceneName-Clean`, and `mmddyy-SceneName-Clutter` using the same `mmddyy-SceneName` name. Copy-and-paste is the safest way to rename folders, while hotkeys such as `F2` can be used for efficiency.

</details>

You can preview sample organized scenes [here](https://drive.google.com/drive/folders/1t0XX_gPOVYVyD8S8YPaTw9M8rLCB0-n7?usp=sharing).

## Preprocessing

<details>
<summary>Directory Structure</summary>
  
```
Root/
├── mmddyy-SceneName
├── process_raw_data.sh
├── check_size.sh
├── make_all_folder_all.sh
├── make_all_folder.sh
├── make_all.py
└── 
```

</details>

Preprocessing
```
# Convert all images to the .JPG extension and normalize image orientation
# Requires ImageMagick, jhead, and ExifTool to be installed on the system
bash process_raw_data.sh

# Ensure all images within a scene have the same resolution.
# If inconsistent image sizes are found, remove or fix them manually.
bash check_size.sh

# Create `mmddyy-SceneName/mmddyy-SceneName-All` folder
bash make_all_folder_all.sh
```

<details>
<summary>Current Directory Structure</summary>
  
```
mmddyy-SceneName/
├── mmddyy-SceneName-All
│   └── images
│       ├── clutter_ImageName.JPG
│       ├── ...
│       └── extra_ImageName.JPG
├── mmddyy-SceneName-Clean
└── mmddyy-SceneName-Clutter
```

</details>

You can preview sample preprocessed scenes [here](https://drive.google.com/drive/folders/1tLgeZCB6ZPJaBAZV4kx5N_bXUmZcraQP?usp=sharing).

## Pose Estimation and Tool-Assisted Verification

COLMAP Pose Estimation and Undistortion
```
# Run COLMAP3.7 pose estimation and undistortion.
# Requires COLMAP.
# If pose estimation fails, remove all generated files in mmddyy-SceneName-All except the images folder,
# then consider tuning Mapper.abs_pose_min_num_inliers or Mapper.min_model_size
# (e.g., reduce them from 35 to 15 and from 20 to 10, respectively).
bash pose_estimation_all.sh
```

<details>
<summary>Current Directory Structure</summary>
  
```
mmddyy-SceneName/
├── mmddyy-SceneName-All
│   ├── images
│   ├── sparse
│   ├── undistortion_images
│   ├── undistortion_sparse
│   ├── split.json
│   ├── transforms.json
│   ├── transforms_clutter.json
│   └── transforms_extra.json
├── mmddyy-SceneName-Clean
└── mmddyy-SceneName-Clutter
```

</details>

You can preview sample processed scenes [here](https://drive.google.com/drive/folders/1hHbe3AkQiEOEI0qgWuznPfnyUMzPQXKf?usp=sharing).

</details>

Tool-Assisted Verification
```
# Requires instant-ngp.
cd instant-ngp

# Visualize the scene using instant-ngp.
# Enable Debug Visualization/Visualize Cameras.
# Check that extra and clutter views are present and that the scene looks correct.
# If not, rerun COLMAP pose estimation with different parameters.
./instant-ngp <path>/mmddyy-SceneName/mmddyy-SceneName-All/transforms.json
```

## Wrap-Up

```
# Create ZIP archives for all processed scene folders.
bash zip_folders.sh
```


