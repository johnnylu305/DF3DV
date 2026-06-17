#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = [".png", ".PNG"]


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def save_rgb(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def extract_right_half(path: Path) -> np.ndarray:
    """Extract the Rendering half from a |GT|Rendering| image."""
    img = load_rgb(path)
    h, w, c = img.shape

    if w % 2 != 0:
        raise ValueError(f"Expected even width for |GT|Rendering| image, got {w}: {path}")

    return img[:, w // 2 :, :]


def list_scene_dirs(parent: Path):
    if not parent.is_dir():
        raise FileNotFoundError(f"Missing folder: {parent}")
    return sorted([p for p in parent.iterdir() if p.is_dir()])


def get_scene_all_dir(scene_dir: Path) -> Path:
    scene_all = scene_dir / f"{scene_dir.name}-All"
    if not scene_all.is_dir():
        raise FileNotFoundError(f"Missing scene-All folder: {scene_all}")
    return scene_all


def collect_star_scenes(root: Path, start: int, end: int):
    """
    Return (chunk_name, scene_dir) for DF3DV-1K-Star.

    Output should keep the chunk level:
      leaderboard/<method>/DF3DV-1K-Star/0000/<scene>/extra_*.png
    """
    star_root = root / "DF3DV-1K-Star"
    if not star_root.is_dir():
        raise FileNotFoundError(f"Missing DF3DV-1K-Star folder: {star_root}")

    scenes = []
    for cid in range(start, end + 1):
        chunk_name = f"{cid:04d}"
        chunk_dir = star_root / chunk_name
        if not chunk_dir.is_dir():
            raise FileNotFoundError(f"Missing chunk folder: {chunk_dir}")

        for scene_dir in list_scene_dirs(chunk_dir):
            scenes.append((chunk_name, scene_dir))

    return scenes


def collect_41_scenes(root: Path):
    """
    Return (None, scene_dir) for DF3DV-41.

    DF3DV-41 does not have chunk folders:
      leaderboard/<method>/DF3DV-41/<scene>/extra_*.png
    """
    df41_root = root / "DF3DV-41"
    if not df41_root.is_dir():
        raise FileNotFoundError(f"Missing DF3DV-41 folder: {df41_root}")

    return [(None, scene_dir) for scene_dir in list_scene_dirs(df41_root)]


def extract_dataset(root: Path, dataset_name: str, scene_entries, method: str):
    out_root = root / "leaderboard" / method / dataset_name

    for chunk_name, scene_dir in tqdm(scene_entries, desc=f"Extract {dataset_name}/{method}"):
        scene_all = get_scene_all_dir(scene_dir)

        renders_dir = scene_all / "MODELS" / method / "renders"
        if not renders_dir.is_dir():
            raise FileNotFoundError(f"Missing renders folder: {renders_dir}")

        render_paths = []
        for ext in IMAGE_EXTS:
            render_paths.extend(renders_dir.glob(f"extra_*{ext}"))

        render_paths = sorted(set(render_paths))
        if len(render_paths) == 0:
            raise FileNotFoundError(f"No extra_*.png found in: {renders_dir}")

        if chunk_name is None:
            scene_out_dir = out_root / scene_dir.name
        else:
            scene_out_dir = out_root / chunk_name / scene_dir.name

        scene_out_dir.mkdir(parents=True, exist_ok=True)

        for render_path in render_paths:
            rendering = extract_right_half(render_path)
            out_path = scene_out_dir / render_path.name
            save_rgb(out_path, rendering)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--eval_star", action="store_true")
    parser.add_argument("--eval_41", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=24)
    args = parser.parse_args()

    root = Path(args.root)

    if not args.eval_star and not args.eval_41:
        raise ValueError("Please specify at least one of --eval_star or --eval_41")

    if args.eval_star:
        scenes = collect_star_scenes(root, args.start, args.end)
        extract_dataset(root, "DF3DV-1K-Star", scenes, args.method)

    if args.eval_41:
        scenes = collect_41_scenes(root)
        extract_dataset(root, "DF3DV-41", scenes, args.method)


if __name__ == "__main__":
    main()
