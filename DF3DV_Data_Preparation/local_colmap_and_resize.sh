#!/bin/bash
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Set to 0 if you do not have a GPU.
USE_GPU=1
# Path to a directory `base/` with images in `base/images/`.
DATASET_PATH=$1
# Recommended CAMERA values: OPENCV for perspective, OPENCV_FISHEYE for fisheye.
CAMERA=${2:-OPENCV}


# Run COLMAP.

### Feature extraction
#colmap feature_extractor \
#    --database_path "$DATASET_PATH"/database.db \
#    --image_path "$DATASET_PATH"/images \
#    --ImageReader.single_camera 1 \
#    --ImageReader.camera_model "$CAMERA" \
#    --SiftExtraction.max_num_features 12000 \
#    --SiftExtraction.use_gpu "$USE_GPU"



# GPU SIFT with 12000
echo "🔹 Trying GPU SIFT with 12000 features..."
if ! colmap feature_extractor \
    --database_path "$DATASET_PATH"/database.db \
    --image_path "$DATASET_PATH"/images \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model "$CAMERA" \
    --SiftExtraction.max_num_features 12000 \
    --SiftExtraction.use_gpu 1; then

    echo "SIFT 12000 failed, retrying with 8000..."

    rm -rf "$DATASET_PATH"/sparse
    rm -rf "$DATASET_PATH"/database.db

    # CPU SIFT with 12000
    colmap feature_extractor \
        --database_path "$DATASET_PATH"/database.db \
        --image_path "$DATASET_PATH"/images \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model "$CAMERA" \
        --SiftExtraction.max_num_features 12000 \
        --SiftExtraction.use_gpu 0
fi

echo "Feature extraction completed successfully."


### Feature matching

colmap exhaustive_matcher \
    --database_path "$DATASET_PATH"/database.db \
    --SiftMatching.use_gpu "$USE_GPU"

## Use if your scene has > 500 images
## Replace this path with your own local copy of the file.
## Download from: https://demuc.de/colmap/#download
# VOCABTREE_PATH=/usr/local/google/home/bmild/vocab_tree_flickr100K_words32K.bin
# colmap vocab_tree_matcher \
#     --database_path "$DATASET_PATH"/database.db \
#     --VocabTreeMatching.vocab_tree_path $VOCABTREE_PATH \
#     --SiftMatching.use_gpu "$USE_GPU"


### Bundle adjustment

# The default Mapper tolerance is unnecessarily large,
# decreasing it speeds up bundle adjustment steps.
# 35, 4, 20
# 0.000001
mkdir -p "$DATASET_PATH"/sparse
colmap mapper \
    --database_path "$DATASET_PATH"/database.db \
    --image_path "$DATASET_PATH"/images \
    --output_path "$DATASET_PATH"/sparse \
    --Mapper.abs_pose_min_num_inliers=35 \
    --Mapper.max_reg_trials=4 \
    --Mapper.min_model_size=20 \
    --Mapper.ba_global_function_tolerance=0.000001


### Image undistortion

## Use this if you want to undistort your images into ideal pinhole intrinsics.
mkdir -p "$DATASET_PATH"/tmp
rm -rf "$DATASET_PATH"/undistortion_images
rm -rf "$DATASET_PATH"/undistortion_sparse/0/
colmap image_undistorter \
     --image_path "$DATASET_PATH"/images \
     --input_path "$DATASET_PATH"/sparse/0 \
     --output_path "$DATASET_PATH"/tmp \
     --output_type COLMAP

## reorganize
mkdir -p "$DATASET_PATH"/undistortion_sparse/0/
mv "$DATASET_PATH"/tmp/sparse/* "$DATASET_PATH"/undistortion_sparse/0/
mv "$DATASET_PATH"/tmp/images "$DATASET_PATH"/undistortion_images

## save space
rm -rf "$DATASET_PATH"/database.db
rm -rf "$DATASET_PATH"/tmp

# Resize images.

#cp -r "$DATASET_PATH"/images "$DATASET_PATH"/images_2

#pushd "$DATASET_PATH"/images_2
#ls | xargs -P 8 -I {} mogrify -resize 50% {}
#popd

#cp -r "$DATASET_PATH"/images "$DATASET_PATH"/images_4

#pushd "$DATASET_PATH"/images_4
#ls | xargs -P 8 -I {} mogrify -resize 25% {}
#popd

#cp -r "$DATASET_PATH"/images "$DATASET_PATH"/images_8

#pushd "$DATASET_PATH"/images_8
#ls | xargs -P 8 -I {} mogrify -resize 12.5% {}
#popd
