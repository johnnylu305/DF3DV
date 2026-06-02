## Face and License Plate Masking
- Download the YOLO [face detection model](https://github.com/akanametov/yolo-face) and [license plate detection model](https://huggingface.co/morsetechlab/yolov11-license-plate-detection).
- Download the Meta folder from Google Drive or Hugging Face (see [here]()).
```
# Blur DF3DV-1K-Star or DF3DV-41
# Example
python blur_face_and_license_plate.py --root 0000 --face_model yolov11l-face.pt --plate_model license-plate-finetune-v1x.pt

# Blur DF3DV-1K-Fixer
# Example
python blur_fixer_images.py --root 0000 --face_model yolov11l-face.pt --plate_model license-plate-finetune-v1x.pt
```
