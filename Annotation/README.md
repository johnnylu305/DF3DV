## Metadata Labeling
```
# Example
python label_meta.py --root 0000 --meta Meta
```


## Face and License Plate Masking
- Download the YOLO [face detection model](https://github.com/akanametov/yolo-face) and [license plate detection model](https://huggingface.co/morsetechlab/yolov11-license-plate-detection).
- Download the Meta folder from Google Drive or Hugging Face (see [here](https://github.com/johnnylu305/DF3DV#df3dv-1k)).
```
# Blur DF3DV-1K-Star or DF3DV-41
# Example
python blur_face_and_license_plate.py --root 0000 --face_model yolov11l-face.pt --plate_model license-plate-finetune-v1x.pt

# Manually remove false face and license plate detections, typically in actor or statue scenes
# Example
python blur_face_and_license_plate_interactive.py --root DF3DV-41 --face_model yolov11l-face.pt --plate_model license-plate-finetune-v1x.pt --interactive

# Blur DF3DV-1K-Fixer
# Example
python blur_fixer_images.py --root 0000 --face_model yolov11l-face.pt --plate_model license-plate-finetune-v1x.pt
```
