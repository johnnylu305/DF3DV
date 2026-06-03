import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Use torchmetrics==1.7.0; newer versions are significantly slower.
# https://github.com/Lightning-AI/torchmetrics/issues/3267
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


PROMPT = "remove degradation"


def load_image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tensor = transforms.ToTensor()(img)  # [0, 1]
    tensor = tensor * 2.0 - 1.0          # LPIPS expects [-1, 1]
    return tensor.unsqueeze(0).to(device)


def get_hw(path: Path):
    img = Image.open(path)
    w, h = img.size
    return h, w


def compute_lpips(source_path: Path, target_path: Path, lpips_metric, device):
    source = load_image_tensor(source_path, device)
    target = load_image_tensor(target_path, device)

    with torch.no_grad():
        score = lpips_metric(source, target)

    return float(score.item())


def parse_series(stem: str, scene: str, method: str) -> str:
    prefix = f"{scene}_{method}_"
    if not stem.startswith(prefix):
        raise ValueError(f"Filename does not match expected format: {stem}")

    return stem[len(prefix):]


def collect_split(
    split_root: Path,
    split_name: str,
    device: torch.device,
    lpips_metric,
):
    results = {}

    for dataset_dir in sorted(split_root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name

        if split_name == "train":
            scene_dirs = []
            for chunk_dir in sorted(dataset_dir.iterdir()):
                if not chunk_dir.is_dir():
                    continue

                chunk = chunk_dir.name
                for scene_dir in sorted(chunk_dir.iterdir()):
                    if scene_dir.is_dir():
                        scene_dirs.append((chunk, scene_dir))
        else:
            scene_dirs = [
                ("", scene_dir)
                for scene_dir in sorted(dataset_dir.iterdir())
                if scene_dir.is_dir()
            ]

        for chunk, scene_dir in scene_dirs:
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

                desc = f"{split_name}/{dataset}"
                if chunk:
                    desc += f"/{chunk}"
                desc += f"/{scene}/{method}"

                for source_path in tqdm(source_files, desc=desc, unit="img"):
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

                    key = base_stem
                    if key in results:
                        raise ValueError(f"Duplicate key found: {key}")

                    h, w = get_hw(source_path)

                    lpips = compute_lpips(
                        source_path,
                        target_path,
                        lpips_metric,
                        device,
                    )

                    series = parse_series(base_stem, scene, method)

                    rel_source = "./" + source_path.relative_to(split_root.parent).as_posix()
                    rel_target = "./" + target_path.relative_to(split_root.parent).as_posix()

                    results[key] = {
                        "image": rel_source,
                        "target_image": rel_target,
                        "ref_image": None,
                        "prompt": PROMPT,
                        "lpips": lpips,
                        "H": h,
                        "W": w,
                        "dataset": dataset,
                        "chunk": chunk,
                        "scene": scene,
                        "method": method,
                        "series": series,
                    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--output", type=str, default="df3dv_fixer_meta.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    root = Path(args.root)
    train_root = root / "Train"
    val_root = root / "Val"

    if not train_root.exists():
        raise FileNotFoundError(f"Missing Train folder: {train_root}")

    if not val_root.exists():
        raise FileNotFoundError(f"Missing Val folder: {val_root}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex",
        normalize=False,
    ).to(device)
    lpips_metric.eval()

    output = {
        "train": collect_split(train_root, "train", device, lpips_metric),
        "test": collect_split(val_root, "test", device, lpips_metric),
    }

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved JSON to: {output_path}")
    print(f"Train samples: {len(output['train'])}")
    print(f"Test samples: {len(output['test'])}")


if __name__ == "__main__":
    main()
