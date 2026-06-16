#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm


GT_EXTS = [".JPG", ".jpg", ".JPEG", ".jpeg", ".PNG", ".png"]


def mean_or_nan(xs: List[float]) -> float:
    return float(np.mean(xs)) if len(xs) > 0 else float("nan")


def load_rgb_float01(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


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
    star_root = root / "DF3DV-1K-Star"
    if not star_root.is_dir():
        raise FileNotFoundError(f"Missing DF3DV-1K-Star folder: {star_root}")

    scenes = []
    for cid in range(start, end + 1):
        chunk_dir = star_root / f"{cid:04d}"
        if not chunk_dir.is_dir():
            raise FileNotFoundError(f"Missing chunk folder: {chunk_dir}")
        scenes.extend(list_scene_dirs(chunk_dir))

    return scenes


def collect_41_scenes(root: Path):
    df41_root = root / "DF3DV-41"
    if not df41_root.is_dir():
        raise FileNotFoundError(f"Missing DF3DV-41 folder: {df41_root}")
    return list_scene_dirs(df41_root)


def collect_gt_images(scene_all: Path):
    gt_dir = scene_all / "undistortion_images_8"
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Missing GT folder: {gt_dir}")

    gt_paths = []
    for ext in GT_EXTS:
        gt_paths.extend(gt_dir.glob(f"extra_*{ext}"))

    gt_paths = sorted(set(gt_paths))
    if len(gt_paths) == 0:
        raise FileNotFoundError(f"No extra_* GT images found in: {gt_dir}")

    return gt_paths


def collect_pairs(root: Path, dataset_name: str, scene_dir: Path, method: str):
    scene_all = get_scene_all_dir(scene_dir)
    gt_paths = collect_gt_images(scene_all)

    pred_dir = root / "leaderboard" / method / dataset_name / scene_dir.name
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Missing prediction folder: {pred_dir}")

    pairs = []
    for gt_path in gt_paths:
        pred_path = pred_dir / f"{gt_path.stem}.png"
        if not pred_path.exists():
            raise FileNotFoundError(
                f"Missing prediction for GT:\n"
                f"GT:   {gt_path}\n"
                f"Pred: {pred_path}"
            )
        pairs.append((gt_path, pred_path))

    return pairs


class PairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        gt_path, pred_path = self.pairs[idx]

        gt = load_rgb_float01(gt_path)
        pred = load_rgb_float01(pred_path)

        if gt.shape != pred.shape:
            raise ValueError(
                f"Shape mismatch:\n"
                f"GT:   {gt_path}, shape={gt.shape}\n"
                f"Pred: {pred_path}, shape={pred.shape}"
            )

        gt = torch.from_numpy(gt).permute(2, 0, 1).float()
        pred = torch.from_numpy(pred).permute(2, 0, 1).float()

        return pred, gt


@torch.no_grad()
def compute_scene_metrics(
    pairs: List[Tuple[Path, Path]],
    device: torch.device,
    num_workers: int,
    psnr_m,
    ssim_m,
    lpips_m,
):
    ds = PairDataset(pairs)
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


def write_placeholder_avg_row(writer: csv.writer, header_len: int):
    writer.writerow(["__DATASET_AVG__"] + [""] * (header_len - 1))


def patch_avg_row(csv_path: Path, avg_row: List[object]):
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")

    with csv_path.open("r", newline="") as rf:
        rows = list(csv.reader(rf))

    rows[1] = [str(x) for x in avg_row]

    with tmp.open("w", newline="") as wf:
        csv.writer(wf).writerows(rows)

    os.replace(tmp, csv_path)


def evaluate_dataset(
    root: Path,
    dataset_name: str,
    scene_dirs,
    method: str,
    device: torch.device,
    num_workers: int,
):
    out_dir = root / "leaderboard" / method
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{dataset_name}_metrics.csv"

    psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_m = LearnedPerceptualImagePatchSimilarity(
        net_type="alex",
        normalize=True,
    ).to(device)

    header = ["object_name", "psnr", "ssim", "lpips"]

    all_psnr, all_ssim, all_lpips = [], [], []

    print(f"[INFO] Evaluating {dataset_name}/{method}")
    print(f"[INFO] CSV: {csv_path}")

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        write_placeholder_avg_row(writer, len(header))

        for scene_dir in tqdm(scene_dirs, desc=f"Benchmark {dataset_name}/{method}"):
            pairs = collect_pairs(root, dataset_name, scene_dir, method)

            psnr, ssim, lpips = compute_scene_metrics(
                pairs=pairs,
                device=device,
                num_workers=num_workers,
                psnr_m=psnr_m,
                ssim_m=ssim_m,
                lpips_m=lpips_m,
            )

            all_psnr.append(psnr)
            all_ssim.append(ssim)
            all_lpips.append(lpips)

            writer.writerow([scene_dir.name, psnr, ssim, lpips])
            f.flush()

            #print(
            #    f"{dataset_name}, {scene_dir.name}, {method}, "
            #    f"PSNR={psnr:.6f}, SSIM={ssim:.6f}, LPIPS={lpips:.6f}"
            #)

    avg_row = [
        "__DATASET_AVG__",
        mean_or_nan(all_psnr),
        mean_or_nan(all_ssim),
        mean_or_nan(all_lpips),
    ]
    patch_avg_row(csv_path, avg_row)

    print(f"[DONE] CSV written: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--eval_star", action="store_true")
    parser.add_argument("--eval_41", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is False.")

    root = Path(args.root)
    device = torch.device("cuda")

    if not args.eval_star and not args.eval_41:
        raise ValueError("Please specify at least one of --eval_star or --eval_41")

    if args.eval_star:
        scenes = collect_star_scenes(root, args.start, args.end)
        evaluate_dataset(
            root=root,
            dataset_name="DF3DV-1K-Star",
            scene_dirs=scenes,
            method=args.method,
            device=device,
            num_workers=args.num_workers,
        )

    if args.eval_41:
        scenes = collect_41_scenes(root)
        evaluate_dataset(
            root=root,
            dataset_name="DF3DV-41",
            scene_dirs=scenes,
            method=args.method,
            device=device,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
