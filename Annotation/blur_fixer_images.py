# This script automatically detects and blurs faces and license plates in
# rendered images using YOLO models. Face and license plate detections are
# performed on the corresponding 3DGS target image and then applied to all
# methods for consistency. The script overwrites the original PNG images and
# generates a JSON file for each image containing the detections and detection
# counts.
# ex: python blur_fixer_images.py --root 0000 --face_model yolov11l-face.pt --plate_model license-plate-finetune-v1x.pt 
from pathlib import Path
import argparse
import json
import cv2
from tqdm import tqdm
from ultralytics import YOLO

# https://github.com/akanametov/yolo-face
# https://huggingface.co/morsetechlab/yolov11-license-plate-detection

IMG_EXTS = {".png", ".PNG"}


def blur_roi(img, xyxy, ksize=99):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return img

    roi = img[y1:y2, x1:x2]

    k = min(
        ksize,
        roi.shape[0] // 2 * 2 + 1,
        roi.shape[1] // 2 * 2 + 1,
    )
    k = max(3, k)

    if k % 2 == 0:
        k += 1

    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return img


def make_padded_box(xyxy, img_shape, pad_ratio):
    h, w = img_shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)

    bw = x2 - x1
    bh = y2 - y1

    px = int(bw * pad_ratio)
    py = int(bh * pad_ratio)

    return [
        max(0, x1 - px),
        max(0, y1 - py),
        min(w, x2 + px),
        min(h, y2 + py),
    ]


def detect_boxes(
    img,
    model,
    det_type,
    conf,
    iou,
    imgsz,
    pad_ratio,
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
        max_det=1000,
    )[0]

    if results.boxes is None:
        return detections

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

        detections.append({
            "type": det_type,
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": score,
            "bbox_xyxy_original": [round(v, 2) for v in xyxy],
            "bbox_xyxy_padded": make_padded_box(
                xyxy=xyxy,
                img_shape=img.shape,
                pad_ratio=pad_ratio,
            ),
        })

    return detections


def apply_detections_to_image(
    img_path,
    detections,
    face_blur_ksize,
    plate_blur_ksize,
):
    img = cv2.imread(str(img_path))

    if img is None:
        return False

    for det in detections:
        if det["type"] == "face":
            img = blur_roi(
                img,
                det["bbox_xyxy_padded"],
                face_blur_ksize,
            )
        elif det["type"] == "license-plate":
            img = blur_roi(
                img,
                det["bbox_xyxy_padded"],
                plate_blur_ksize,
            )

    cv2.imwrite(str(img_path), img)
    return True


def save_meta(
    img_path,
    root,
    meta_root,
    detection_source,
    scene_name,
    method_name,
    detections,
    blur_success,
):
    rel = img_path.relative_to(root)
    meta_path = meta_root / rel.with_suffix(".json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "image": str(img_path),
        "detections": detections,
        "num_faces": sum(d["type"] == "face" for d in detections),
        "num_license_plates": sum(
            d["type"] == "license-plate" for d in detections
        ),
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def collect_3dgs_targets(root):
    items = []

    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue

        scene_name = scene_dir.name
        models_dir = scene_dir / f"{scene_name}-All" / "MODELS"
        d3gs_dir = models_dir / "3DGS"

        if not d3gs_dir.exists():
            continue

        for p in sorted(d3gs_dir.glob("*_target.png")):
            items.append((scene_dir, models_dir, p))

    return items


def get_all_method_dirs(models_dir):
    if not models_dir.exists():
        return []

    return sorted([p for p in models_dir.iterdir() if p.is_dir()])


def make_method_filename(d3gs_filename, method_name):
    if "_3DGS_" in d3gs_filename:
        return d3gs_filename.replace("_3DGS_", f"_{method_name}_")

    return d3gs_filename


def get_source_target_paths(method_dir, d3gs_target_path):
    method_name = method_dir.name

    target_name = make_method_filename(
        d3gs_target_path.name,
        method_name,
    )

    source_name = target_name.replace("_target.png", "_source.png")

    return [
        method_dir / target_name,
        method_dir / source_name,
    ]


def process_one_3dgs_target(
    scene_dir,
    models_dir,
    d3gs_target_path,
    root,
    meta_root,
    face_model,
    plate_model,
    args,
):
    img = cv2.imread(str(d3gs_target_path))

    if img is None:
        print(f"Warning: cannot read {d3gs_target_path}")
        return 0

    detections = []

    if face_model is not None:
        detections.extend(detect_boxes(
            img=img,
            model=face_model,
            det_type="face",
            conf=args.face_conf,
            iou=args.iou,
            imgsz=args.imgsz,
            pad_ratio=args.face_pad_ratio,
            min_w=args.face_min_w,
            min_h=args.face_min_h,
        ))

    if plate_model is not None:
        detections.extend(detect_boxes(
            img=img,
            model=plate_model,
            det_type="license-plate",
            conf=args.plate_conf,
            iou=args.iou,
            imgsz=args.imgsz,
            pad_ratio=args.plate_pad_ratio,
            min_w=args.plate_min_w,
            min_h=args.plate_min_h,
            aspect_min=args.plate_aspect_min,
            aspect_max=args.plate_aspect_max,
        ))

    saved_count = 0
    # apply the masks to all methods
    method_dirs = get_all_method_dirs(models_dir)

    for method_dir in method_dirs:
        method_name = method_dir.name

        for img_path in get_source_target_paths(method_dir, d3gs_target_path):
            if not img_path.exists():
                continue

            ok = apply_detections_to_image(
                img_path=img_path,
                detections=detections,
                face_blur_ksize=args.face_blur_ksize,
                plate_blur_ksize=args.plate_blur_ksize,
            )

            save_meta(
                img_path=img_path,
                root=root,
                meta_root=meta_root,
                detection_source=d3gs_target_path,
                scene_name=scene_dir.name,
                method_name=method_name,
                detections=detections,
                blur_success=ok,
            )

            saved_count += 1

    return saved_count


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="0000")

    parser.add_argument("--face_model", type=str, default=None)
    parser.add_argument("--plate_model", type=str, default=None)

    parser.add_argument("--face_conf", type=float, default=0.75) # tune confidence threshold for face detection
    parser.add_argument("--plate_conf", type=float, default=0.62) # tune confidence threshold for license plate detection
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=512) # tune this if OOM

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
    meta_root = root.parent / f"{root.name}-MF"

    face_model = YOLO(args.face_model) if args.face_model else None
    plate_model = YOLO(args.plate_model) if args.plate_model else None

    # use the 3DGS target image for face and license plate detection.
    items = collect_3dgs_targets(root)

    print(f"Found {len(items)} 3DGS target images.")
    print(f"Metadata root: {meta_root}")

    total_meta = 0

    for scene_dir, models_dir, d3gs_target_path in tqdm(items):
        total_meta += process_one_3dgs_target(
            scene_dir=scene_dir,
            models_dir=models_dir,
            d3gs_target_path=d3gs_target_path,
            root=root,
            meta_root=meta_root,
            face_model=face_model,
            plate_model=plate_model,
            args=args,
        )

    print("Done.")
    print(f"Total PNG metas saved: {total_meta}")
    print(f"Blurred images replaced in: {root}")
    print(f"Metadata saved in: {meta_root}")


if __name__ == "__main__":
    main()
