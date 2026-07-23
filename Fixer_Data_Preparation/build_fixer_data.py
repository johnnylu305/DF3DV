#!/usr/bin/env python3
import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


METHODS = [
    "3DGS",
    "ASYMGS",
    "DEGS",
    "DESPLAT",
    "OCSPLAT",
    "RBSPLAT",
    "SLS",
    "T3DGS",
    "T3DGS_TMR",
    "WILDGS",
]

IMAGE_EXTS = [".png", ".PNG"]


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def save_rgb(path: Path, img: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def split_gt_rendering(path: Path):
    img = load_rgb(path)
    _, w, _ = img.shape

    if w % 2 != 0:
        raise ValueError(f"Expected even width for |GT|Rendering| image, got {w}: {path}")

    gt = img[:, : w // 2, :]
    rendering = img[:, w // 2 :, :]
    return gt, rendering


def list_scene_dirs(parent: Path):
    if not parent.is_dir():
        raise FileNotFoundError(f"Missing folder: {parent}")
    return sorted([p for p in parent.iterdir() if p.is_dir()])


def get_scene_all_dir(scene_dir: Path) -> Path:
    scene_all = scene_dir / f"{scene_dir.name}-All"
    if not scene_all.is_dir():
        raise FileNotFoundError(f"Missing scene-All folder: {scene_all}")
    return scene_all


def collect_star_scenes(src_root: Path, start: int, end: int):
    star_root = src_root / "DF3DV-1K-Star"
    if not star_root.is_dir():
        raise FileNotFoundError(f"Missing DF3DV-1K-Star folder: {star_root}")

    items = []
    for chunk_id in range(start, end + 1):
        chunk_name = f"{chunk_id:04d}"
        chunk_dir = star_root / chunk_name

        if not chunk_dir.is_dir():
            raise FileNotFoundError(f"Missing chunk folder: {chunk_dir}")

        for scene_dir in list_scene_dirs(chunk_dir):
            items.append((chunk_name, scene_dir))

    return items


def collect_41_scenes(src_root: Path):
    df41_root = src_root / "DF3DV-41"
    if not df41_root.is_dir():
        raise FileNotFoundError(f"Missing DF3DV-41 folder: {df41_root}")

    return [(None, scene_dir) for scene_dir in list_scene_dirs(df41_root)]


def copy_undistortion_sparse(src_scene_all: Path, dst_scene_all: Path, overwrite: bool) -> None:
    src_sparse = src_scene_all / "undistortion_sparse"
    dst_sparse = dst_scene_all / "undistortion_sparse"

    if not src_sparse.is_dir():
        raise FileNotFoundError(f"Missing undistortion_sparse folder: {src_sparse}")

    if dst_sparse.exists():
        if not overwrite:
            raise FileExistsError(f"Output undistortion_sparse already exists: {dst_sparse}")
        shutil.rmtree(dst_sparse)

    dst_scene_all.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_sparse, dst_sparse)


def collect_render_paths(renders_dir: Path):
    if not renders_dir.is_dir():
        raise FileNotFoundError(f"Missing renders folder: {renders_dir}")

    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(renders_dir.glob(f"extra_*{ext}"))

    paths = sorted(set(paths))

    if len(paths) == 0:
        raise FileNotFoundError(f"No extra_*.png found in: {renders_dir}")

    return paths


def convert_scene_to_fixer(
    src_scene_dir: Path,
    dst_scene_dir: Path,
    method: str,
    overwrite: bool,
) -> str:
    src_scene_all = get_scene_all_dir(src_scene_dir)
    dst_scene_all = dst_scene_dir / f"{src_scene_dir.name}-All"

    src_renders_dir = src_scene_all / "MODELS" / method / "renders"
    render_paths = collect_render_paths(src_renders_dir)

    dst_method_dir = dst_scene_all / "MODELS" / method

    if dst_method_dir.exists() and not overwrite:
        raise FileExistsError(f"Output method folder already exists: {dst_method_dir}")

    dst_method_dir.mkdir(parents=True, exist_ok=True)

    copy_undistortion_sparse(src_scene_all, dst_scene_all, overwrite)

    for render_path in render_paths:
        gt, rendering = split_gt_rendering(render_path)

        source_name = f"{src_scene_dir.name}_{method}_{render_path.stem}_source.png"
        target_name = f"{src_scene_dir.name}_{method}_{render_path.stem}_target.png"

        save_rgb(dst_method_dir / source_name, rendering, overwrite=overwrite)
        save_rgb(dst_method_dir / target_name, gt, overwrite=overwrite)

    return str(src_scene_dir)


def run_parallel_jobs(jobs, num_workers: int, desc: str):
    if num_workers <= 1:
        for job in tqdm(jobs, desc=desc):
            convert_scene_to_fixer(*job)
        return

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(convert_scene_to_fixer, *job) for job in jobs]

        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            future.result()


def convert_star(
    src_root: Path,
    dst_root: Path,
    method: str,
    start: int,
    end: int,
    overwrite: bool,
    num_workers: int,
) -> None:
    items = collect_star_scenes(src_root, start, end)

    jobs = []
    for chunk_name, src_scene_dir in items:
        dst_scene_dir = (
            dst_root
            / "Train"
            / "DF3DV-1K-Star-Fixer"
            / chunk_name
            / src_scene_dir.name
        )
        jobs.append((src_scene_dir, dst_scene_dir, method, overwrite))

    run_parallel_jobs(
        jobs,
        num_workers=num_workers,
        desc=f"Convert Star/{method}",
    )


def convert_41(
    src_root: Path,
    dst_root: Path,
    method: str,
    overwrite: bool,
    num_workers: int,
) -> None:
    items = collect_41_scenes(src_root)

    jobs = []
    for _, src_scene_dir in items:
        dst_scene_dir = (
            dst_root
            / "Val"
            / "DF3DV-41-Fixer"
            / src_scene_dir.name
        )
        jobs.append((src_scene_dir, dst_scene_dir, method, overwrite))

    run_parallel_jobs(
        jobs,
        num_workers=num_workers,
        desc=f"Convert DF3DV-41/{method}",
    )


def resolve_methods(args_methods):
    if len(args_methods) == 0:
        return METHODS

    unknown = [m for m in args_methods if m not in METHODS]
    if len(unknown) > 0:
        raise ValueError(
            f"Unknown methods: {unknown}\n"
            f"Allowed methods: {METHODS}"
        )

    return args_methods


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_root", type=str, required=True)
    parser.add_argument("--dst_root", type=str, required=True)

    parser.add_argument(
        "--methods",
        nargs="*",
        default=[],
        help="Methods to process. Default: all methods.",
    )

    parser.add_argument("--run_star", action="store_true")
    parser.add_argument("--run_41", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    methods = resolve_methods(args.methods)

    if not args.run_star and not args.run_41:
        raise ValueError("Please specify at least one of --run_star or --run_41")

    print(f"[INFO] Source root: {src_root}")
    print(f"[INFO] Destination root: {dst_root}")
    print(f"[INFO] Methods: {methods}")
    print(f"[INFO] Run Star: {args.run_star}")
    print(f"[INFO] Run DF3DV-41: {args.run_41}")
    print(f"[INFO] Overwrite: {args.overwrite}")
    print(f"[INFO] Num workers: {args.num_workers}")

    for method in methods:
        print(f"\n[INFO] Processing method: {method}")

        if args.run_star:
            convert_star(
                src_root=src_root,
                dst_root=dst_root,
                method=method,
                start=args.start,
                end=args.end,
                overwrite=args.overwrite,
                num_workers=args.num_workers,
            )

        if args.run_41:
            convert_41(
                src_root=src_root,
                dst_root=dst_root,
                method=method,
                overwrite=args.overwrite,
                num_workers=args.num_workers,
            )


if __name__ == "__main__":
    main()
