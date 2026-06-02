# This script automatically detects and blurs faces and license plates in a
# DF3DV-style dataset using YOLO models. License plate detection is only
# performed for outdoor scenes. The script overwrites the original JPG images
# by re-saving the blurred results and generates a JSON file for each image
# containing the detections and detection counts. Since JPG is a lossy format,
# re-saved images may exhibit minor compression differences.
# ex: python blur_face_and_license_plate.py --root 0000 --face_model yolov11l-face.pt --plate_model license-plate-finetune-v1x.pt 
from pathlib import Path
import argparse
import json
import cv2
from tqdm import tqdm
from ultralytics import YOLO


# https://github.com/akanametov/yolo-face
# https://huggingface.co/morsetechlab/yolov11-license-plate-detection

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def blur_roi(img, xyxy, pad_ratio=0.30, ksize=99):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)

    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_ratio), int(bh * pad_ratio)

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)

    if x2 <= x1 or y2 <= y1:
        return img, None

    roi = img[y1:y2, x1:x2]

    k = min(ksize, roi.shape[0] // 2 * 2 + 1, roi.shape[1] // 2 * 2 + 1)
    k = max(3, k)
    if k % 2 == 0:
        k += 1

    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return img, [x1, y1, x2, y2]


def run_detector(
    img,
    model,
    det_type,
    conf,
    iou,
    imgsz,
    pad_ratio,
    blur_ksize,
    min_w,
    min_h,
    aspect_min=None,
    aspect_max=None,
):
    detections = []

    results = model.predict(
        source=img,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
        max_det=1000
    )[0]

    if results.boxes is None:
        return img, detections

    names = results.names

    for box in results.boxes:
        cls_id = int(box.cls.item())
        cls_name = names.get(cls_id, str(cls_id))
        score = float(box.conf.item())
        xyxy = box.xyxy[0].cpu().numpy().tolist()

        bw = xyxy[2] - xyxy[0]
        bh = xyxy[3] - xyxy[1]

        if bw < min_w or bh < min_h:
            continue

        aspect = bw / max(bh, 1e-6)
        if aspect_min is not None and aspect < aspect_min:
            continue
        if aspect_max is not None and aspect > aspect_max:
            continue

        img, padded_box = blur_roi(
            img,
            xyxy,
            pad_ratio=pad_ratio,
            ksize=blur_ksize
        )

        if padded_box is not None:
            detections.append({
                "type": det_type,
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": score,
                "bbox_xyxy_original": [round(v, 2) for v in xyxy],
                "bbox_xyxy_padded": padded_box
            })

    return img, detections


def load_scene_meta(scene_dir, scene_meta_root):
    scene_name = scene_dir.name

    candidates = [
        scene_meta_root / f"{scene_name}_meta.json",
        scene_meta_root / f"{scene_name}.json",
    ]

    for meta_path in candidates:
        if meta_path.exists():
            with open(meta_path, "r") as f:
                return json.load(f), meta_path

    return {}, None


def is_outdoor_scene(scene_meta):
    return str(scene_meta.get("environment", "")).lower() == "outdoor"


def should_run_face_blur(img_path, folder_type):
    """
    Face blur rule:

    1. For All folders:
       - clutter_* images need face blur.
       - extra_* / clean_* images do not need face blur.

    2. For Clean and Clutter folders:
       - filenames usually do not contain clutter_* or extra_*.
       - all images should run face blur.
    """
    if folder_type == "All":
        stem_lower = img_path.stem.lower()
        return stem_lower.startswith("clutter")

    return True


def process_image(
    img_path,
    meta_path,
    folder_type,
    image_folder,
    face_model,
    plate_model,
    run_plate_blur,
    scene_meta,
    scene_meta_path,
    face_conf,
    plate_conf,
    iou,
    imgsz,
    face_pad_ratio,
    plate_pad_ratio,
    face_blur_ksize,
    plate_blur_ksize,
    face_min_w,
    face_min_h,
    plate_min_w,
    plate_min_h,
    plate_aspect_min,
    plate_aspect_max,
):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Warning: cannot read {img_path}")
        return

    all_detections = []

    run_face_blur = should_run_face_blur(img_path, folder_type)

    if face_model is not None and run_face_blur:
        img, face_dets = run_detector(
            img=img,
            model=face_model,
            det_type="face",
            conf=face_conf,
            iou=iou,
            imgsz=imgsz,
            pad_ratio=face_pad_ratio,
            blur_ksize=face_blur_ksize,
            min_w=face_min_w,
            min_h=face_min_h,
            aspect_min=None,
            aspect_max=None,
        )
        all_detections.extend(face_dets)

    if plate_model is not None and run_plate_blur:
        img, plate_dets = run_detector(
            img=img,
            model=plate_model,
            det_type="license-plate",
            conf=plate_conf,
            iou=iou,
            imgsz=imgsz,
            pad_ratio=plate_pad_ratio,
            blur_ksize=plate_blur_ksize,
            min_w=plate_min_w,
            min_h=plate_min_h,
            aspect_min=plate_aspect_min,
            aspect_max=plate_aspect_max,
        )
        all_detections.extend(plate_dets)

    metadata = {
        "image": str(img_path),
        "detections": all_detections,
        "num_faces": sum(d["type"] == "face" for d in all_detections),
        "num_license_plates": sum(d["type"] == "license-plate" for d in all_detections),
    }

    cv2.imwrite(str(img_path), img)

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def collect_images(root, scene_meta_root):
    image_items = []

    # we will apply masks to images in these folders
    folder_specs = [
        ("All", "images"),
        ("All", "undistortion_images"),
        ("Clean", "images"),
        ("Clutter", "images"),
    ]

    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue

        scene_name = scene_dir.name
        
        # load meta data    
        scene_meta, scene_meta_path = load_scene_meta(scene_dir, scene_meta_root)
        # we only detect license plates in outdoor scenes
        run_plate_blur = is_outdoor_scene(scene_meta)

        if scene_meta_path is None:
            #print(f"Warning: missing scene meta for {scene_name}; plate blur disabled.")
            raise FileNotFoundError(
                f"Missing scene metadata for scene '{scene_name}'. "
                f"Expected either "
                f"'{scene_meta_root / (scene_name + '_meta.json')}' "
                f"or "
                f"'{scene_meta_root / (scene_name + '.json')}'."
            )

        for folder_type, image_folder in folder_specs:
            img_dir = scene_dir / f"{scene_name}-{folder_type}" / image_folder
            if not img_dir.exists():
                continue

            for p in sorted(img_dir.iterdir()):
                if p.suffix in IMG_EXTS:
                    image_items.append(
                        (
                            p,
                            folder_type,
                            image_folder,
                            run_plate_blur,
                            scene_meta,
                            scene_meta_path,
                        )
                    )

    return image_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="0000")

    parser.add_argument("--scene_meta_root", type=str, default=None)

    parser.add_argument("--face_model", type=str, default=None)
    parser.add_argument("--plate_model", type=str, default=None)

    parser.add_argument("--face_conf", type=float, default=0.62) # tune confidence threshold for face detection
    parser.add_argument("--plate_conf", type=float, default=0.62) # tune confidence threshold for license plate detection
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=3040) # tune this if OOM

    parser.add_argument("--face_pad_ratio", type=float, default=0.0)
    parser.add_argument("--plate_pad_ratio", type=float, default=0.0)

    parser.add_argument("--face_blur_ksize", type=int, default=150)
    parser.add_argument("--plate_blur_ksize", type=int, default=100)

    parser.add_argument("--face_min_w", type=float, default=1)
    parser.add_argument("--face_min_h", type=float, default=1)

    parser.add_argument("--plate_min_w", type=float, default=2)
    parser.add_argument("--plate_min_h", type=float, default=2)
    parser.add_argument("--plate_aspect_min", type=float, default=1.00001)
    parser.add_argument("--plate_aspect_max", type=float, default=10.0)

    args = parser.parse_args()

    if args.face_model is None and args.plate_model is None:
        raise ValueError("Please provide at least --face_model or --plate_model")

    root = Path(args.root)
    meta_root = root.parent / f"{root.name}-M"

    if args.scene_meta_root is not None:
        scene_meta_root = Path(args.scene_meta_root)
    else:
        scene_meta_root = root.parent / "Meta" / root.name

    face_model = YOLO(args.face_model) if args.face_model else None
    plate_model = YOLO(args.plate_model) if args.plate_model else None

    image_items = collect_images(root, scene_meta_root)

    print(f"Found {len(image_items)} images.")
    print(f"Scene metadata root: {scene_meta_root}")
    print(f"Metadata will be saved in: {meta_root}")

    for (
        img_path,
        folder_type,
        image_folder,
        run_plate_blur,
        scene_meta,
        scene_meta_path,
    ) in tqdm(image_items):

        rel = img_path.relative_to(root)
        meta_path = meta_root / rel.with_suffix(".json")

        process_image(
            img_path=img_path,
            meta_path=meta_path,
            folder_type=folder_type,
            image_folder=image_folder,
            face_model=face_model,
            plate_model=plate_model,
            run_plate_blur=run_plate_blur,
            scene_meta=scene_meta,
            scene_meta_path=scene_meta_path,
            face_conf=args.face_conf,
            plate_conf=args.plate_conf,
            iou=args.iou,
            imgsz=args.imgsz,
            face_pad_ratio=args.face_pad_ratio,
            plate_pad_ratio=args.plate_pad_ratio,
            face_blur_ksize=args.face_blur_ksize,
            plate_blur_ksize=args.plate_blur_ksize,
            face_min_w=args.face_min_w,
            face_min_h=args.face_min_h,
            plate_min_w=args.plate_min_w,
            plate_min_h=args.plate_min_h,
            plate_aspect_min=args.plate_aspect_min,
            plate_aspect_max=args.plate_aspect_max,
        )

    print("Done.")
    print(f"Blurred images replaced in: {root}")
    print(f"Metadata saved in: {meta_root}")


if __name__ == "__main__":
    main()
