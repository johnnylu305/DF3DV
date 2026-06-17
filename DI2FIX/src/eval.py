"""
Streamed, per-dataset CSV (sheet-style) + per-image printing.

AVERAGING POLICY:
- Per-scene metric for a method = mean over images in that scene.
- Dataset average for a method = mean over SCENES (scene-balanced),
  i.e. mean of the per-scene averages (NOT weighted by #images).

Folder structure:
  <root>/
    <method>/
      <dataset>/
        <scene>/
          <scene-All>/
            <images_dir>/*.png   (grid images)

Grid layout assumed 2x3:
  Row0: V0 GT | V0 Input | V0 Fixed
  Row1: V1 GT | V1 Input | V1 Fixed (ignored)

Modes:
  --mode fixed : compare V0 Fixed vs V0 GT
  --mode input : compare V0 Input vs V0 GT

Outputs:
  <out_dir>/<dataset>_<mode>_scene_metrics.csv
  - First row after header: __DATASET_AVG__ (patched at end)
  - Then one row per scene: object_name, <method>_psnr/<method>_ssim/<method>_lpips

Console:
  - Optional per-image: dataset, method, scene, image, psnr, ssim, lpips
  - Always per-scene avg lines + dataset avg lines
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from di2fix_utils import (
    list_dirs,
    find_scene_all,
    collect_grid_images,
    extract_v0_pred_gt,
    np_to_torch_chw,
    metrics_one_pair,
    mean_or_nan,
    write_placeholder_avg_row,
    rewrite_first_data_row,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root folder containing method folders.")
    ap.add_argument("--mode", type=str, required=True, choices=["fixed", "input"])
    ap.add_argument("--images_dir", type=str, default="images")
    ap.add_argument("--out_dir", type=str, default="", help="Default: <root>/metrics_csv")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--print_per_image", action="store_true", help="Print per-image line with method included.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    out_dir = Path(args.out_dir) if args.out_dir else (root / "metrics_csv_alex")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.device == "cpu" or not torch.cuda.is_available() else "cuda")

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    methods = list_dirs(root)
    if not methods:
        raise RuntimeError(f"No method folders under {root}")

    dataset_map: Dict[str, Dict[str, Path]] = {}
    for method_dir in methods:
        method = method_dir.name
        for dataset_dir in list_dirs(method_dir):
            dataset_map.setdefault(dataset_dir.name, {})[method] = dataset_dir

    for dataset, method_to_datasetdir in sorted(dataset_map.items(), key=lambda x: x[0]):
        method_names = sorted(method_to_datasetdir.keys())

        header = ["object_name"]
        for m in method_names:
            header += [f"{m}_psnr", f"{m}_ssim", f"{m}_lpips"]

        csv_path = out_dir / f"{dataset}_{args.mode}_scene_metrics.csv"

        ds_scene_avgs: Dict[str, Dict[str, List[float]]] = {
            m: {"psnr": [], "ssim": [], "lpips": []} for m in method_names
        }

        all_scenes = set()
        for m, ddir in method_to_datasetdir.items():
            for scene_dir in list_dirs(ddir):
                all_scenes.add(scene_dir.name)
        all_scenes = sorted(all_scenes)

        print("\n" + "=" * 80)
        print(f"DATASET: {dataset} | mode={args.mode} | methods={len(method_names)} | scenes={len(all_scenes)}")
        print("=" * 80)

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            write_placeholder_avg_row(writer, header)
            f.flush()

            for scene in all_scenes:
                out_row: Dict[str, object] = {"object_name": scene}
                for m in method_names:
                    out_row[f"{m}_psnr"] = ""
                    out_row[f"{m}_ssim"] = ""
                    out_row[f"{m}_lpips"] = ""

                for method in method_names:
                    dataset_dir = method_to_datasetdir[method]
                    scene_dir = dataset_dir / scene
                    if not scene_dir.is_dir():
                        continue

                    scene_all = find_scene_all(scene_dir)
                    if scene_all is None:
                        continue

                    grid_paths = collect_grid_images(scene_all, args.images_dir)
                    if not grid_paths:
                        continue

                    psnrs, ssims, lpipss = [], [], []
                    for gp in grid_paths:
                        pred_np, gt_np = extract_v0_pred_gt(gp, mode=args.mode)
                        pred = np_to_torch_chw(pred_np, device=device)
                        gt = np_to_torch_chw(gt_np, device=device)

                        ps, ss, lp = metrics_one_pair(pred, gt, psnr_metric, ssim_metric, lpips_metric)
                        psnrs.append(ps)
                        ssims.append(ss)
                        lpipss.append(lp)

                        if args.print_per_image:
                            print(f"{dataset}, {method}, {scene}, {gp.name}, {ps:.6f}, {ss:.6f}, {lp:.6f}")

                    scene_ps = mean_or_nan(psnrs)
                    scene_ss = mean_or_nan(ssims)
                    scene_lp = mean_or_nan(lpipss)

                    out_row[f"{method}_psnr"] = scene_ps
                    out_row[f"{method}_ssim"] = scene_ss
                    out_row[f"{method}_lpips"] = scene_lp

                    if not np.isnan(scene_ps):
                        ds_scene_avgs[method]["psnr"].append(scene_ps)
                    if not np.isnan(scene_ss):
                        ds_scene_avgs[method]["ssim"].append(scene_ss)
                    if not np.isnan(scene_lp):
                        ds_scene_avgs[method]["lpips"].append(scene_lp)

                    print(
                        f"[SCENE_AVG] {dataset}, {method}, {scene}: "
                        f"psnr={scene_ps:.6f}, ssim={scene_ss:.6f}, lpips={scene_lp:.6f}"
                    )

                row_list: List[object] = [out_row["object_name"]]
                for m in method_names:
                    row_list += [out_row[f"{m}_psnr"], out_row[f"{m}_ssim"], out_row[f"{m}_lpips"]]
                writer.writerow(row_list)
                f.flush()

        avg_row: List[object] = ["__DATASET_AVG__"]
        print("\n--- DATASET_AVG (across scenes) ---")
        for m in method_names:
            ds_ps = mean_or_nan(ds_scene_avgs[m]["psnr"])
            ds_ss = mean_or_nan(ds_scene_avgs[m]["ssim"])
            ds_lp = mean_or_nan(ds_scene_avgs[m]["lpips"])
            avg_row += [ds_ps, ds_ss, ds_lp]
            print(f"{dataset}, {m}: psnr={ds_ps:.6f}, ssim={ds_ss:.6f}, lpips={ds_lp:.6f}")

        rewrite_first_data_row(csv_path, avg_row)
        print(f"[OK] wrote (streamed) {csv_path}")


if __name__ == "__main__":
    main()
