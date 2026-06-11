## Downscale Images

Downscale images by a factor of 8, which is a commonly used downsampling factor for distractor-free radiance fields.

```bash
# Create undistortion_images_8 from undistortion_images
python downsample_images.py --root DF3DV-1K --factor 8 --num_workers 8 --overwrite
