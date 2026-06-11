# Downsample Images for DF3DV-1K
# 
# This script creates downsampled copies of scene images for the DF3DV-1K dataset.
# https://github.com/johnnylu305/SpotLessSplats-gendf/blob/gendf_undist/examples/datasets/colmap.py
# A small issue is that the saved images are actually in PNG format but use a .JPG extension.
# But anyway, this is what SLS does as well.
# For each scene:
#
#     undistortion_images/
#         ->
#     undistortion_images_<factor>/
# 
# Supported subsets:
#     - DF3DV-1K-Star
#         DF3DV-1K-Star/<chunk>/<scene>/<scene>-All/undistortion_images
#     - DF3DV-41
#         DF3DV-41/<scene>/<scene>-All/undistortion_images
# 
# Features:
#     - Parallel image processing via ProcessPoolExecutor
#     - Configurable downsampling factor (--factor)
#     - Configurable number of workers (--num_workers)
#     - Optional overwrite of existing outputs (--overwrite)
#     - Raises errors when expected dataset folders are missing
# 
# Example:
#     python downsample_images.py --root DF3DV-1K --factor 8 --num_workers 8 --overwrite
import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mediapy as media
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# change chunks list
STAR_CHUNKS = [f"{i:04d}" for i in range(25)]


def is_image(path: Path) -> bool:
    return path.suffix in IMAGE_EXTS


def downscale_image(src_path: Path, dst_path: Path, factor: int):
    img = media.read_image(src_path)

    new_h = img.shape[0] // factor
    new_w = img.shape[1] // factor

    if new_h <= 0 or new_w <= 0:
        raise ValueError(f"Image too small for factor {factor}: {src_path}")

    resized = media.resize_image(img, (new_h, new_w))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    media.write_image(dst_path, resized)


def _downscale_worker(args):
    src_path, dst_path, factor = args
    downscale_image(src_path, dst_path, factor)
    return 1


def downscale_folder(src_dir: Path, dst_dir: Path, factor: int, overwrite: bool, num_workers: int):
    if not src_dir.exists():
        raise FileNotFoundError(f"Missing source folder: {src_dir}")

    if dst_dir.exists():
        if overwrite:
            shutil.rmtree(dst_dir)
        else:
            print(f"[Skip] Output exists: {dst_dir}")
            return

    images = sorted([p for p in src_dir.iterdir() if p.is_file() and is_image(p)])

    if len(images) == 0:
        raise RuntimeError(f"No images found in source folder: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(src_path, dst_dir / src_path.name, factor) for src_path in images]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(
            tqdm(
                executor.map(_downscale_worker, jobs),
                total=len(jobs),
                desc=f"Downscale {src_dir.name}",
                leave=False,
            )
        )


def process_scene(scene_dir: Path, factor: int, overwrite: bool, num_workers: int):
    scene_name = scene_dir.name
    scene_all_dir = scene_dir / f"{scene_name}-All"

    src_dir = scene_all_dir / "undistortion_images"
    dst_dir = scene_all_dir / f"undistortion_images_{factor}"

    if not src_dir.exists():
        raise FileNotFoundError(f"Missing source folder: {src_dir}")

    downscale_folder(src_dir, dst_dir, factor, overwrite, num_workers)


def process_star(root: Path, factor: int, overwrite: bool, num_workers: int):
    star_root = root / "DF3DV-1K-Star"

    if not star_root.exists():
        raise FileNotFoundError(f"Missing subset folder: {star_root}")

    for chunk in tqdm(STAR_CHUNKS, desc="DF3DV-1K-Star chunks"):
        chunk_dir = star_root / chunk

        if not chunk_dir.exists():
            raise FileNotFoundError(f"Missing chunk folder: {chunk_dir}")

        scene_dirs = sorted([p for p in chunk_dir.iterdir() if p.is_dir()])

        if len(scene_dirs) == 0:
            raise RuntimeError(f"No scene folders found in chunk: {chunk_dir}")

        for scene_dir in tqdm(scene_dirs, desc=f"Scenes {chunk}", leave=False):
            process_scene(scene_dir, factor, overwrite, num_workers)


def process_df3dv41(root: Path, factor: int, overwrite: bool, num_workers: int):
    val_root = root / "DF3DV-41"

    if not val_root.exists():
        raise FileNotFoundError(f"Missing subset folder: {val_root}")

    scene_dirs = sorted([p for p in val_root.iterdir() if p.is_dir()])

    if len(scene_dirs) == 0:
        raise RuntimeError(f"No scene folders found in: {val_root}")

    for scene_dir in tqdm(scene_dirs, desc="DF3DV-41 scenes"):
        process_scene(scene_dir, factor, overwrite, num_workers)


def main():
    parser = argparse.ArgumentParser(
        description="Create downscaled undistortion_images_factor folders using mediapy."
    )
    parser.add_argument("--root", type=str, required=True, help="Dataset root, e.g. DF3DV-1K")
    parser.add_argument("--factor", type=int, default=8, help="Downscale factor. Default: 8")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output folders.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers. Default: 8")
    args = parser.parse_args()

    if args.factor <= 1:
        raise ValueError("--factor should be larger than 1")

    if args.num_workers <= 0:
        raise ValueError("--num_workers should be larger than 0")

    root = Path(args.root)

    if not root.exists():
        raise FileNotFoundError(f"Missing dataset root: {root}")

    process_star(root, args.factor, args.overwrite, args.num_workers)
    process_df3dv41(root, args.factor, args.overwrite, args.num_workers)

    print("Done.")


if __name__ == "__main__":
    main()
