# This script automatically detects faces and license plates in a DF3DV-style
# dataset using YOLO models and optionally supports interactive review using
# PIL visualization. Users can manually remove incorrect detections before
# blurring is applied. The script overwrites the original JPG images with the
# final blurred results and generates a JSON file for each image containing
# the detections and detection counts. Since JPG is a lossy format, re-saved
# images may exhibit minor compression differences. We typically use this
# script to remove the faces of actors appearing in scenes, such as those in
# the Actor scenario.
# ex: python blur_face_and_license_plate_interactive.py --root DF3DV-41 --face_model yolov11l-face.pt --plate_model license-plate-finetune-v1x.pt --interactive
from pathlib import Path
import argparse
import json
import cv2
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image

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


def detect_objects(
    img,
    model,
    det_type,
    conf,
    iou,
    imgsz,
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
        })

    return detections


def apply_blur_to_detections(
    img,
    detections,
    face_pad_ratio,
    plate_pad_ratio,
    face_blur_ksize,
    plate_blur_ksize,
):
    kept_detections = []

    for det in detections:
        if det["type"] == "face":
            pad_ratio = face_pad_ratio
            blur_ksize = face_blur_ksize
        elif det["type"] == "license-plate":
            pad_ratio = plate_pad_ratio
            blur_ksize = plate_blur_ksize
        else:
            continue

        img, padded_box = blur_roi(
            img,
            det["bbox_xyxy_original"],
            pad_ratio=pad_ratio,
            ksize=blur_ksize,
        )

        if padded_box is not None:
            det = dict(det)
            det["bbox_xyxy_padded"] = padded_box
            kept_detections.append(det)

    return img, kept_detections


def draw_numbered_bboxes(img, detections):
    vis = img.copy()

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = map(int, det["bbox_xyxy_original"])

        color = (0, 255, 0) if det["type"] == "face" else (0, 165, 255)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

        label = f"{idx}: {det['type']} {det['confidence']:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        y_text_top = max(0, y1 - th - baseline - 8)
        y_text_base = y_text_top + th + 4

        cv2.rectangle(
            vis,
            (x1, y_text_top),
            (
                min(vis.shape[1] - 1, x1 + tw + 8),
                min(vis.shape[0] - 1, y_text_top + th + baseline + 8),
            ),
            color,
            -1,
        )

        cv2.putText(
            vis,
            label,
            (x1 + 4, y_text_base),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    return vis


def resize_for_display(img, max_side=1600):
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return img

    return cv2.resize(
        img,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def show_preview_with_pil(vis_img, max_side=1600):
    show_img = resize_for_display(vis_img, max_side=max_side)
    img_pil = Image.fromarray(cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB))
    img_pil.show()


def parse_remove_input(text, num_detections):
    text = text.strip().lower()

    if text in {"", "nom", "none", "no", "n"}:
        return set()

    if text == "all":
        return set(range(num_detections))

    text = text.replace(",", " ")
    remove_ids = set()

    for token in text.split():
        if not token.isdigit():
            raise ValueError(
                f"Invalid token: {token}. Please input like: 1, 3, 5 or nom"
            )

        idx = int(token)
        if idx < 0 or idx >= num_detections:
            raise ValueError(
                f"Index {idx} out of range. Valid range: 0 to {num_detections - 1}"
            )

        remove_ids.add(idx)

    return remove_ids


def interactive_filter_detections(
    img_path,
    img,
    detections,
    max_display_side,
):
    if len(detections) == 0:
        print(f"{img_path}: no detections.")
        return detections, []

    vis = draw_numbered_bboxes(img, detections)
    show_preview_with_pil(vis, max_side=max_display_side)

    print("\nDetections:")
    for idx, det in enumerate(detections):
        print(
            f"  {idx}: {det['type']}, conf={det['confidence']:.3f}, "
            f"bbox={det['bbox_xyxy_original']}"
        )

    while True:
        remove_text = input(
            f"Remove bbox numbers for {img_path.name} "
            "(e.g. 1, 3, 5; use 'nom' for no remove; 'all' to remove all): "
        )

        try:
            remove_ids = parse_remove_input(remove_text, len(detections))
            break
        except ValueError as e:
            print(f"Invalid input: {e}")

    kept = [det for i, det in enumerate(detections) if i not in remove_ids]
    removed = [det for i, det in enumerate(detections) if i in remove_ids]

    print(f"Kept {len(kept)} / {len(detections)} detections.")
    return kept, removed


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
    interactive,
    max_display_side,
):
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        print(f"Warning: cannot read {img_path}")
        return

    all_detections = []
    run_face_blur = should_run_face_blur(img_path, folder_type)

    if face_model is not None and run_face_blur:
        all_detections.extend(detect_objects(
            img=original_img,
            model=face_model,
            det_type="face",
            conf=face_conf,
            iou=iou,
            imgsz=imgsz,
            min_w=face_min_w,
            min_h=face_min_h,
        ))

    if plate_model is not None and run_plate_blur:
        all_detections.extend(detect_objects(
            img=original_img,
            model=plate_model,
            det_type="license-plate",
            conf=plate_conf,
            iou=iou,
            imgsz=imgsz,
            min_w=plate_min_w,
            min_h=plate_min_h,
            aspect_min=plate_aspect_min,
            aspect_max=plate_aspect_max,
        ))

    if interactive:
        kept_detections, _ = interactive_filter_detections(
            img_path=img_path,
            img=original_img,
            detections=all_detections,
            max_display_side=max_display_side,
        )
    else:
        kept_detections = all_detections

    updated_img = original_img.copy()
    updated_img, kept_detections = apply_blur_to_detections(
        img=updated_img,
        detections=kept_detections,
        face_pad_ratio=face_pad_ratio,
        plate_pad_ratio=plate_pad_ratio,
        face_blur_ksize=face_blur_ksize,
        plate_blur_ksize=plate_blur_ksize,
    )

    metadata = {
        "image": str(img_path),
        "detections": kept_detections,
        "num_faces": sum(d["type"] == "face" for d in kept_detections),
        "num_license_plates": sum(
            d["type"] == "license-plate" for d in kept_detections
        ),
    }

    cv2.imwrite(str(img_path), updated_img)

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def collect_images(root, scene_meta_root):
    image_items = []

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

        scene_meta, scene_meta_path = load_scene_meta(scene_dir, scene_meta_root)
        run_plate_blur = is_outdoor_scene(scene_meta)

        if scene_meta_path is None:
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

    parser.add_argument("--face_conf", type=float, default=0.65)
    parser.add_argument("--plate_conf", type=float, default=0.99)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=3040)

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

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Review detections interactively using PIL visualization.",
    )
    parser.add_argument(
        "--max_display_side",
        type=int,
        default=1600,
        help="Max side length for PIL preview display.",
    )

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
    print(f"Interactive mode: {args.interactive}")
    print("Input examples: nom | 1, 3, 5 | all")

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
            interactive=args.interactive,
            max_display_side=args.max_display_side,
        )

    print("Done.")
    print(f"Blurred images replaced in: {root}")
    print(f"Metadata saved in: {meta_root}")


if __name__ == "__main__":
    main()
