# Create DF3DV-1K-Fixer / DF3DV-41-Fixer dataset JSON.
#
# The script traverses all scenes under Train/ and Val/, finds paired
# *_source.png and *_target.png images under:
#
#   <scene>-All/MODELS/<method>/
#
# For each image pair, it:
#   1. Computes LPIPS (AlexNet backbone).
#   2. Records image resolution (H, W).
#   3. Stores dataset, chunk, scene, method, and series metadata.
#   4. Exports a JSON file for training and evaluation.
#
# A PyTorch Dataset + DataLoader is used for parallel image loading.
# Since images from different scenes may have different resolutions,
# the DataLoader uses batch_size=1 and multiple workers.
#
# Progress is reported at group level:
#   Train: one log per chunk
#   Val:   one log per dataset
#
# Sample key format:
#   Train: {dataset}_{chunk}_{base_stem}
#   Val:   {dataset}_{base_stem}
#
# ex: python create_fixer_dataset_json.py --root DF3DV-1K-Fixer --output dataset_df3dv1k.json --num_workers 8


import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Use torchmetrics==1.7.0; newer versions are significantly slower.
# https://github.com/Lightning-AI/torchmetrics/issues/3267
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


PROMPT = "remove degradation"


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        tensor = transforms.ToTensor()(img)  # [0, 1]
        tensor = tensor * 2.0 - 1.0          # LPIPS expects [-1, 1]
    return tensor


def get_hw(path: Path):
    with Image.open(path) as img:
        w, h = img.size
    return h, w


def parse_series(stem: str, scene: str, method: str) -> str:
    prefix = f"{scene}_{method}_"

    if not stem.startswith(prefix):
        raise ValueError(f"Filename does not match expected format: {stem}")

    return stem[len(prefix):]


def collect_scene_samples(
    split_root: Path,
    split_name: str,
    dataset: str,
    chunk: str,
    scene_dir: Path,
):
    samples = []

    scene = scene_dir.name
    all_dir = scene_dir / f"{scene}-All"
    models_dir = all_dir / "MODELS"

    if not models_dir.exists():
        raise FileNotFoundError(f"Missing MODELS folder: {models_dir}")

    for method_dir in sorted(models_dir.iterdir()):
        if not method_dir.is_dir():
            continue

        method = method_dir.name
        source_files = sorted(method_dir.glob("*_source.png"))

        for source_path in source_files:
            stem = source_path.stem

            if not stem.endswith("_source"):
                raise ValueError(f"Unexpected source filename: {source_path}")

            base_stem = stem[:-len("_source")]
            target_path = source_path.with_name(base_stem + "_target.png")

            if not target_path.exists():
                raise FileNotFoundError(
                    f"Missing target image for source:\n"
                    f"source: {source_path}\n"
                    f"target: {target_path}"
                )

            # Different chunks may contain images with the same base_stem,
            # so include the chunk in the key for training samples.
            if split_name == "train":
                key = f"{dataset}_{chunk}_{base_stem}"
            else:
                key = f"{dataset}_{base_stem}"

            h, w = get_hw(source_path)
            series = parse_series(base_stem, scene, method)

            rel_source = "./" + source_path.relative_to(split_root.parent).as_posix()
            rel_target = "./" + target_path.relative_to(split_root.parent).as_posix()

            samples.append(
                {
                    "key": key,
                    "source_path": source_path,
                    "target_path": target_path,
                    "image": rel_source,
                    "target_image": rel_target,
                    "ref_image": None,
                    "prompt": PROMPT,
                    "H": h,
                    "W": w,
                    "dataset": dataset,
                    "chunk": chunk,
                    "scene": scene,
                    "method": method,
                    "series": series,
                }
            )

    return samples


class FixerPairDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]

        source = load_image_tensor(meta["source_path"])
        target = load_image_tensor(meta["target_path"])

        return {
            "source": source,
            "target": target,
            "meta": meta,
        }


def collate_fn(batch):
    """
    Custom collate function for batch_size=1.

    The default PyTorch collate function cannot handle pathlib.Path objects
    inside meta. We keep meta as a normal Python dict and only add the batch
    dimension to source/target tensors.
    """
    if len(batch) != 1:
        raise ValueError("This script assumes batch_size=1.")

    item = batch[0]

    return {
        "source": item["source"].unsqueeze(0),
        "target": item["target"].unsqueeze(0),
        "meta": item["meta"],
    }


def compute_samples_lpips(
    samples,
    device: torch.device,
    lpips_metric,
    num_workers: int,
):
    dataset = FixerPairDataset(samples)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    results = {}

    for batch in loader:
        source = batch["source"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        meta = batch["meta"]
        key = meta["key"]

        if key in results:
            raise ValueError(f"Duplicate key found: {key}")

        with torch.no_grad():
            lpips_score = lpips_metric(source, target)

        lpips = float(lpips_score.item())

        results[key] = {
            "image": meta["image"],
            "target_image": meta["target_image"],
            "ref_image": None,
            "prompt": meta["prompt"],
            "lpips": lpips,
            "H": int(meta["H"]),
            "W": int(meta["W"]),
            "dataset": meta["dataset"],
            "chunk": meta["chunk"],
            "scene": meta["scene"],
            "method": meta["method"],
            "series": meta["series"],
        }

    return results


def merge_results(global_results, new_results):
    for key, value in new_results.items():
        if key in global_results:
            raise ValueError(f"Duplicate key found: {key}")
        global_results[key] = value


def collect_train(
    train_root: Path,
    device: torch.device,
    lpips_metric,
    num_workers: int,
):
    results = {}

    for dataset_dir in sorted(train_root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name

        for chunk_dir in sorted(dataset_dir.iterdir()):
            if not chunk_dir.is_dir():
                continue

            chunk = chunk_dir.name
            group_start = time.time()

            print(
                f"\n[Start] train/{dataset}/{chunk} at {now_str()}",
                flush=True,
            )

            chunk_samples = []

            for scene_dir in sorted(chunk_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue

                scene_samples = collect_scene_samples(
                    split_root=train_root,
                    split_name="train",
                    dataset=dataset,
                    chunk=chunk,
                    scene_dir=scene_dir,
                )

                chunk_samples.extend(scene_samples)

            print(
                f"[Info] train/{dataset}/{chunk}: {len(chunk_samples)} samples found",
                flush=True,
            )

            chunk_results = compute_samples_lpips(
                samples=chunk_samples,
                device=device,
                lpips_metric=lpips_metric,
                num_workers=num_workers,
            )

            merge_results(results, chunk_results)

            elapsed = time.time() - group_start

            print(
                f"[Done] train/{dataset}/{chunk} finished at {now_str()} "
                f"(elapsed {format_seconds(elapsed)}, samples {len(chunk_results)})",
                flush=True,
            )

    return results


def collect_val(
    val_root: Path,
    device: torch.device,
    lpips_metric,
    num_workers: int,
):
    results = {}

    for dataset_dir in sorted(val_root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name
        group_start = time.time()

        print(
            f"\n[Start] val/{dataset} at {now_str()}",
            flush=True,
        )

        dataset_samples = []

        for scene_dir in sorted(dataset_dir.iterdir()):
            if not scene_dir.is_dir():
                continue

            scene_samples = collect_scene_samples(
                split_root=val_root,
                split_name="val",
                dataset=dataset,
                chunk="",
                scene_dir=scene_dir,
            )

            dataset_samples.extend(scene_samples)

        print(
            f"[Info] val/{dataset}: {len(dataset_samples)} samples found",
            flush=True,
        )

        dataset_results = compute_samples_lpips(
            samples=dataset_samples,
            device=device,
            lpips_metric=lpips_metric,
            num_workers=num_workers,
        )

        merge_results(results, dataset_results)

        elapsed = time.time() - group_start

        print(
            f"[Done] val/{dataset} finished at {now_str()} "
            f"(elapsed {format_seconds(elapsed)}, samples {len(dataset_results)})",
            flush=True,
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--output", type=str, default="df3dv_fixer_meta.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.root)
    train_root = root / "Train"
    val_root = root / "Val"

    if not train_root.exists():
        raise FileNotFoundError(f"Missing Train folder: {train_root}")

    if not val_root.exists():
        raise FileNotFoundError(f"Missing Val folder: {val_root}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[Info] Using device : {device}", flush=True)
    print("[Info] Batch size   : 1", flush=True)
    print(f"[Info] Num workers  : {args.num_workers}", flush=True)

    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex",
        normalize=False,
    ).to(device)
    lpips_metric.eval()

    start_time = time.time()

    output = {
        "train": collect_train(
            train_root=train_root,
            device=device,
            lpips_metric=lpips_metric,
            num_workers=args.num_workers,
        ),
        "test": collect_val(
            val_root=val_root,
            device=device,
            lpips_metric=lpips_metric,
            num_workers=args.num_workers,
        ),
    }

    output_path = Path(args.output)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - start_time

    print(f"\n[Done] Saved JSON to: {output_path}", flush=True)
    print(f"[Done] Train samples: {len(output['train'])}", flush=True)
    print(f"[Done] Test samples : {len(output['test'])}", flush=True)
    print(f"[Done] Total elapsed: {format_seconds(elapsed)}", flush=True)


if __name__ == "__main__":
    main()
