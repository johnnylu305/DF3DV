#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

# hardcode method list
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


IMAGE_EXTS = [".JPG", ".jpg", ".JPEG", ".jpeg", ".PNG", ".png"]


def mean_or_nan(xs: List[float]) -> float:
    return float(np.mean(xs)) if len(xs) > 0 else float("nan")


def find_existing_image(path_no_ext: Path) -> Path:
    for ext in IMAGE_EXTS:
        p = path_no_ext.with_suffix(ext)
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing image: {path_no_ext} with extensions {IMAGE_EXTS}")


def load_rgb_float01(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def extract_rendering_from_concat(path: Path) -> np.ndarray:
    img = load_rgb_float01(path)

    h, w, c = img.shape
    if w % 2 != 0:
        raise ValueError(f"Expected |GT|Rendering| image width to be even, got {w}: {path}")

    return img[:, w // 2 :, :]


def np_to_torch(img: np.ndarray, device: torch.device) -> Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(
        device=device, dtype=torch.float32
    )


class MetricPairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        gt_path, render_path = self.pairs[idx]

        gt = load_rgb_float01(gt_path)
        pred = extract_rendering_from_concat(render_path)

        if gt.shape != pred.shape:
            raise ValueError(
                f"Shape mismatch:\n"
                f"GT:   {gt_path}, shape={gt.shape}\n"
                f"Pred: {render_path}, shape={pred.shape}"
            )

        gt = torch.from_numpy(gt).permute(2, 0, 1).float()
        pred = torch.from_numpy(pred).permute(2, 0, 1).float()

        return pred, gt


@torch.no_grad()
def compute_pairs_metrics(
    pairs: List[Tuple[Path, Path]],
    device: torch.device,
    num_workers: int,
    psnr_m: PeakSignalNoiseRatio,
    ssim_m: StructuralSimilarityIndexMeasure,
    lpips_m: LearnedPerceptualImagePatchSimilarity,
) -> Tuple[float, float, float]:
    if len(pairs) == 0:
        return float("nan"), float("nan"), float("nan")

    ds = MetricPairDataset(pairs)
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    psnrs, ssims, lpipss = [], [], []

    for pred, gt in dl:
        pred = pred.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        psnrs.append(float(psnr_m(pred, gt).item()))
        ssims.append(float(ssim_m(pred, gt).item()))
        lpipss.append(float(lpips_m(pred, gt).item()))

    return mean_or_nan(psnrs), mean_or_nan(ssims), mean_or_nan(lpipss)


def get_scene_all_dir(scene_dir: Path) -> Path:
    scene_all = scene_dir / f"{scene_dir.name}-All"
    if not scene_all.is_dir():
        raise FileNotFoundError(f"Missing scene-All folder: {scene_all}")
    return scene_all


def list_scene_dirs(parent: Path) -> List[Path]:
    if not parent.is_dir():
        raise FileNotFoundError(f"Missing folder: {parent}")
    return sorted([p for p in parent.iterdir() if p.is_dir()])


def collect_gt_images(scene_all: Path) -> List[Path]:
    gt_dir = scene_all / "undistortion_images_8"
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Missing GT folder: {gt_dir}")

    gt_paths = []
    for ext in IMAGE_EXTS:
        gt_paths.extend(gt_dir.glob(f"extra_*{ext}"))

    gt_paths = sorted(set(gt_paths))
    if len(gt_paths) == 0:
        raise FileNotFoundError(f"No extra_* GT images found in: {gt_dir}")

    return gt_paths


def collect_pairs_for_method(scene_all: Path, method: str) -> List[Tuple[Path, Path]]:
    gt_paths = collect_gt_images(scene_all)

    models_dir = scene_all / "MODELS"
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Missing MODELS folder: {models_dir}")

    method_dir = models_dir / method
    if not method_dir.is_dir():
        raise FileNotFoundError(f"Missing method folder: {method_dir}")

    renders_dir = method_dir / "renders"
    if not renders_dir.is_dir():
        raise FileNotFoundError(f"Missing renders folder: {renders_dir}")

    pairs = []
    for gt_path in gt_paths:
        render_stem = gt_path.stem
        render_path = renders_dir / f"{render_stem}.png"

        if not render_path.exists():
            raise FileNotFoundError(
                f"Missing rendering for GT:\n"
                f"GT:     {gt_path}\n"
                f"Render: {render_path}"
            )

        pairs.append((gt_path, render_path))

    return pairs


def write_placeholder_avg_row(writer: csv.writer, header_len: int) -> None:
    writer.writerow(["__DATASET_AVG__"] + [""] * (header_len - 1))


def patch_avg_row(csv_path: Path, avg_row: List[object]) -> None:
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")

    with csv_path.open("r", newline="") as rf:
        rows = list(csv.reader(rf))

    rows[1] = [str(x) if x is not None else "" for x in avg_row]

    with tmp.open("w", newline="") as wf:
        csv.writer(wf).writerows(rows)

    os.replace(tmp, csv_path)


def evaluate_scenes(
    dataset_name: str,
    scene_dirs: List[Path],
    root: Path,
    methods: List[str],
    device: torch.device,
    num_workers: int,
):
    csv_path = root / f"{dataset_name}_metrics.csv"

    psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_m = LearnedPerceptualImagePatchSimilarity(
        net_type="alex",
        normalize=True,
    ).to(device)

    header = ["object_name"]
    for method in methods:
        header += [f"{method}_psnr", f"{method}_ssim", f"{method}_lpips"]

    dataset_avgs: Dict[str, Dict[str, List[float]]] = {
        m: {"psnr": [], "ssim": [], "lpips": []} for m in methods
    }

    print(f"[INFO] Evaluating {dataset_name}")
    print(f"[INFO] CSV: {csv_path}")

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        write_placeholder_avg_row(writer, len(header))

        for scene_dir in tqdm(scene_dirs, desc=f"Scenes({dataset_name})"):
            scene_all = get_scene_all_dir(scene_dir)

            row_vals = {"object_name": scene_dir.name}
            for method in methods:
                row_vals[f"{method}_psnr"] = ""
                row_vals[f"{method}_ssim"] = ""
                row_vals[f"{method}_lpips"] = ""

            for method in tqdm(methods, desc=f"Methods({scene_dir.name})", leave=False):
                pairs = collect_pairs_for_method(scene_all, method)

                psnr, ssim, lpips = compute_pairs_metrics(
                    pairs=pairs,
                    device=device,
                    num_workers=num_workers,
                    psnr_m=psnr_m,
                    ssim_m=ssim_m,
                    lpips_m=lpips_m,
                )

                row_vals[f"{method}_psnr"] = psnr
                row_vals[f"{method}_ssim"] = ssim
                row_vals[f"{method}_lpips"] = lpips

                dataset_avgs[method]["psnr"].append(psnr)
                dataset_avgs[method]["ssim"].append(ssim)
                dataset_avgs[method]["lpips"].append(lpips)

                #print(
                #    f"{dataset_name}, {scene_dir.name}, {method}, "
                #    f"PSNR={psnr:.6f}, SSIM={ssim:.6f}, LPIPS={lpips:.6f}"
                #)

            writer.writerow(
                [row_vals["object_name"]]
                + [
                    row_vals[f"{method}_{metric}"]
                    for method in methods
                    for metric in ("psnr", "ssim", "lpips")
                ]
            )
            f.flush()

    avg_row = ["__DATASET_AVG__"]
    for method in methods:
        avg_row += [
            mean_or_nan(dataset_avgs[method]["psnr"]),
            mean_or_nan(dataset_avgs[method]["ssim"]),
            mean_or_nan(dataset_avgs[method]["lpips"]),
        ]

    patch_avg_row(csv_path, avg_row)

    print(f"[DONE] {dataset_name}: {csv_path}")


def build_star_scene_dirs(root: Path, start: int, end: int) -> List[Path]:
    star_root = root / "DF3DV-1K-Star"
    if not star_root.is_dir():
        raise FileNotFoundError(f"Missing DF3DV-1K-Star folder: {star_root}")

    scene_dirs = []
    for chunk_id in range(start, end + 1):
        chunk_dir = star_root / f"{chunk_id:04d}"
        if not chunk_dir.is_dir():
            raise FileNotFoundError(f"Missing chunk folder: {chunk_dir}")

        scene_dirs.extend(list_scene_dirs(chunk_dir))

    return scene_dirs


def build_41_scene_dirs(root: Path) -> List[Path]:
    df41_root = root / "DF3DV-41"
    if not df41_root.is_dir():
        raise FileNotFoundError(f"Missing DF3DV-41 folder: {df41_root}")

    return list_scene_dirs(df41_root)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--eval_star", action="store_true")
    parser.add_argument("--eval_41", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--exclude_methods", nargs="*", default=[])

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is False.")

    device = torch.device("cuda")

    root = Path(args.root)

    methods = [m for m in METHODS if m not in set(args.exclude_methods)]

    if len(methods) == 0:
        raise ValueError("No methods left after applying --exclude_methods.")

    if not args.eval_star and not args.eval_41:
        raise ValueError("Please specify at least one of: --eval_star or --eval_41")

    print(f"[INFO] Root: {root}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Methods: {methods}")
    print(f"[INFO] Excluded methods: {args.exclude_methods}")

    if args.eval_star:
        star_scene_dirs = build_star_scene_dirs(root, args.start, args.end)
        evaluate_scenes(
            dataset_name="DF3DV-1K-Star",
            scene_dirs=star_scene_dirs,
            root=root,
            methods=methods,
            device=device,
            num_workers=args.num_workers,
        )

    if args.eval_41:
        df41_scene_dirs = build_41_scene_dirs(root)
        evaluate_scenes(
            dataset_name="DF3DV-41",
            scene_dirs=df41_scene_dirs,
            root=root,
            methods=methods,
            device=device,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
