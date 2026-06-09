import sys, json
from pathlib import Path
from pycolmap import SceneManager

def build_split_from_colmap(scene_dir: Path) -> None:
    """Create $SCENE/split.json using pycolmap on $SCENE/undistortion_sparse/0."""
    model_dir = scene_dir / "undistortion_sparse" / "0"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model dir not found: {model_dir}")

    sm = SceneManager(str(model_dir))
    sm.load_cameras()
    sm.load_images()  # points3D not needed

    image_names = [img.name for img in sm.images.values()]
    train = sorted([n for n in image_names if "clutter" in n.lower()])
    test  = sorted([n for n in image_names if "extra"   in n.lower()])

    out_fp = scene_dir / "split.json"
    out_fp.write_text(json.dumps({"train": train, "test": test}, indent=2) + "\n", encoding="utf-8")
    print(f"[split] source={model_dir}")
    print(f"[split] wrote {out_fp}  (train={len(train)} | test={len(test)})")

def main():
    if len(sys.argv) < 2:
        print("usage: make_split_from_colmap_bin.py /path/to/SCENE", file=sys.stderr)
        sys.exit(1)
    build_split_from_colmap(Path(sys.argv[1]))

if __name__ == "__main__":
    main()
