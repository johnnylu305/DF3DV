import torch
import torch.nn.functional as F
import csv
import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def make_vis_grid(x_src, x_tgt_pred, x_tgt):
    """
    Stitch visualization for qualitative evaluation.
    x_src, x_tgt_pred, x_tgt: (b, v, c, h, w)
    Layout (2 × 3 grid):
        src_tar | src | src_output
        ref_tar | ref | ref_output
    """
    b, v, c, h, w = x_src.shape
    # split source/reference views
    # input
    src, ref = x_src[:, 0:1], x_src[:, 1:2]
    # ground truth
    src_tar, ref_tar = x_tgt[:, 0:1], x_tgt[:, 1:2]
    # prediction
    src_output, ref_output = x_tgt_pred[:, 0:1], x_tgt_pred[:, 1:2]

    # row1: src_tar | src | src_output
    row1 = torch.cat([src_tar, src, src_output], dim=-1)   # (b, v, c, h, 3*w)

    # row2: ref_tar | ref | ref_output
    row2 = torch.cat([ref_tar, ref, ref_output], dim=-1)     # (b, v, c, h, 3*w)
    # 2 × 3 grid
    grid = torch.cat([row1, row2], dim=-2)                 # (b, v, c, 2*h, 3*w)
    return grid


def get_psnr(gt, pred):
    """
    Compute PSNR between two images.
    Args:
        gt:   Ground-truth image tensor in [-1, 1].
        pred: Predicted image tensor in [-1, 1].
    Returns:
        PSNR value (dB) averaged over all pixels/channels.
    """
    # convert from [-1, 1] to [0, 1]
    gt = gt * 0.5 + 0.5
    pred = pred * 0.5 + 0.5

    # mean squared error over all elements
    mse = F.mse_loss(pred.float(), gt.float(), reduction="mean")

    # avoid log(0) for identical images
    mse = torch.clamp(mse, min=1e-10)

    # PSNR = 10 * log10(MAX^2 / MSE), where MAX = 1
    return (10 * torch.log10(1.0 / mse)).item()


def compute_metrics(x_pred, x_tgt, net_lpips):
    """
    Compute image quality metrics between prediction and ground truth.

    Args:
        x_pred: Predicted image tensor of shape (1, C, H, W) in [-1, 1].
        x_tgt: Ground-truth image tensor of shape (1, C, H, W) in [-1, 1].
        net_lpips: LPIPS network.

    Returns:
        Dictionary containing:
            - l2: Mean squared error (MSE)
            - lpips: LPIPS perceptual distance
            - psnr: Peak signal-to-noise ratio (dB)
    """
    # compute pixel-wise mean squared error
    loss_l2 = F.mse_loss(
        x_pred.float(),
        x_tgt.float(),
        reduction="mean",
    )

    # compute perceptual similarity (lower is better)
    loss_lpips = net_lpips(
        x_pred.float(),
        x_tgt.float(),
    ).mean()

    # compute PSNR in dB
    psnr = get_psnr(x_tgt, x_pred)

    return {
        "l2": loss_l2.item(),
        "lpips": loss_lpips.item(),
        "psnr": psnr,
    }


def to_uint8(img: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor in range [-1,1] to uint8 [0,255].
    Accepts CHW or BCHW tensors.
    """
    img = img.clamp(-1, 1)         # step 1: clip
    img = (img + 1.0) / 2.0        # step 2: rescale to [0,1]
    img = (img * 255).round()      # step 3: scale to [0,255]
    return img.to(torch.uint8)


def load_pipe_weights_into_model(pipe, model, report: bool = True):
    """
    Load weights from a DifixPipeline (pipe) into a Difix model (model).
    Args:
        pipe: DifixPipeline instance (HuggingFace Diffusers style).
        model: Difix model instance (github training version).
        report: If True, print a summary of loaded/missing/unexpected keys.
    """

    results = {}

    def load_component(dst_module, src_module, name):
        dst_sd = dst_module.state_dict()
        src_sd = src_module.state_dict()
        # filter out mismatched shapes
        src_sd = {
            k: v for k, v in src_sd.items()
            if k in dst_sd and v.shape == dst_sd[k].shape
        }

        missing, unexpected = dst_module.load_state_dict(src_sd, strict=False)
        results[name] = {"missing": missing, "unexpected": unexpected}

        if report:
            if not missing:
                status = "✅ OK"
            else:
                status = "❌ NOT OK"
            print(f"{name}: {dst_module.__class__.__name__} "
                  f"(params: {len(dst_sd)}) --> {status}")
            if missing:
                print(f"   Missing ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"   Unexpected ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
    # UNet
    load_component(model.unet, pipe.unet, "unet")
    # VAE
    load_component(model.vae, pipe.vae, "vae")
    # Text encoder
    load_component(model.text_encoder, pipe.text_encoder, "text_encoder")

    # Copy over tokenizer / scheduler (not modules)
    if hasattr(model, "tokenizer"):
        model.tokenizer = pipe.tokenizer
    if hasattr(model, "scheduler"):
        model.scheduler = pipe.scheduler
    return results


def load_finetuned_ckpt_model_only(model, ckpt_path):
    """
    Load a finetuned DI2FIX checkpoint saved by save_ckpt().

    Loads:
      - state_dict_unet -> model.unet
      - state_dict_vae  -> model.vae (lora+skip connection)

    The optimizer state is ignored because this function is used for inference.
    """
    sd = torch.load(ckpt_path, map_location="cpu")

    if "state_dict_unet" not in sd or "state_dict_vae" not in sd:
        raise ValueError(
            f"Bad checkpoint format. Expected keys: "
            f"'state_dict_unet' and 'state_dict_vae'. "
            f"Got keys: {list(sd.keys())[:20]}"
        )

    missing_u, unexpected_u = model.unet.load_state_dict(
        sd["state_dict_unet"],
        strict=True,
    )

    missing_v, unexpected_v = model.vae.load_state_dict(
        sd["state_dict_vae"],
        strict=False,
    )

    print(
        f"[CKPT] UNet: missing={len(missing_u)} unexpected={len(unexpected_u)} | "
        f"VAE: missing={len(missing_v)} unexpected={len(unexpected_v)}"
    )


def load_rgb_float01(path: Path) -> np.ndarray:
    """
    Load an image as float32 RGB in [0,1].
    """
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def crop_tile(
    grid_rgb: np.ndarray,
    row: int,
    col: int,
    nrows: int = 2,
    ncols: int = 3,
) -> np.ndarray:
    """
    Extract a single tile from a regularly partitioned image grid.

    The grid is assumed to be divided into nrows × ncols cells.
    """
    H, W, _ = grid_rgb.shape

    tile_h = H // nrows
    tile_w = W // ncols

    y0 = row * tile_h
    x0 = col * tile_w

    y1 = (row + 1) * tile_h if row < nrows - 1 else H
    x1 = (col + 1) * tile_w if col < ncols - 1 else W

    return grid_rgb[y0:y1, x0:x1, :]


def extract_v0_pred_gt(
    grid_path: Path,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the V0 prediction and corresponding ground-truth image
    from a saved 2×3 visualization grid.

    Grid layout:
        Row0: V0 GT | V0 Input | V0 Fixed
        Row1: V1 GT | V1 Input | V1 Fixed

    Only the first view (V0) is used for evaluation.

    Args:
        mode:
            "input" -> compare V0 Input vs V0 GT
            "fixed" -> compare V0 Fixed vs V0 GT
    """
    grid = load_rgb_float01(grid_path)

    gt = crop_tile(grid, row=0, col=0)

    if mode == "input":
        pred = crop_tile(grid, row=0, col=1)
    elif mode == "fixed":
        pred = crop_tile(grid, row=0, col=2)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    h = min(pred.shape[0], gt.shape[0])
    w = min(pred.shape[1], gt.shape[1])

    return pred[:h, :w, :], gt[:h, :w, :]


def np_to_torch_chw(
    img: np.ndarray,
    device: torch.device,
) -> Tensor:
    """
    Convert an HWC numpy image in [0,1] to a BCHW float32 tensor.

    Output shape:
        (1, C, H, W)

    This matches the input format expected by TorchMetrics.
    """
    return (
        torch.from_numpy(img)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )


def list_dirs(p: Path) -> List[Path]:
    """
    Return all immediate child directories sorted alphabetically.

    Sorting ensures deterministic traversal and reproducible CSV output.
    """
    return sorted([x for x in p.iterdir() if x.is_dir()])


def find_scene_all(scene_dir: Path) -> Path:
    """
    Locate the scene-All folder corresponding to a scene.

    Raises:
        FileNotFoundError: if the expected scene-All folder does not exist.
    """
    scene_all = scene_dir / f"{scene_dir.name}-All"

    if not scene_all.is_dir():
        raise FileNotFoundError(
            f"Missing scene-All folder:\n"
            f"  scene_dir : {scene_dir}\n"
            f"  expected  : {scene_all}"
        )

    return scene_all


def collect_grid_images(
    scene_all_dir: Path,
    images_dirname: str,
) -> List[Path]:
    """
    Collect all PNG visualization grids under a scene-All image folder.
    """
    img_dir = scene_all_dir / images_dirname

    if not img_dir.is_dir():
        raise FileNotFoundError(
            f"Missing image directory: {img_dir}"
        )

    grid_paths = sorted(img_dir.glob("*.png"))

    if len(grid_paths) == 0:
        raise RuntimeError(
            f"No PNG files found in: {img_dir}"
        )

    return grid_paths


@torch.no_grad()
def metrics_one_pair(
    pred: Tensor,
    gt: Tensor,
    psnr_metric: PeakSignalNoiseRatio,
    ssim_metric: StructuralSimilarityIndexMeasure,
    lpips_metric: LearnedPerceptualImagePatchSimilarity,
) -> Tuple[float, float, float]:
    """
    Compute PSNR, SSIM, and LPIPS for a single image pair.

    Inputs are expected to be BCHW float tensors in [0,1].

    Returns:
        (psnr, ssim, lpips)
    """
    return (
        float(psnr_metric(pred, gt).item()),
        float(ssim_metric(pred, gt).item()),
        float(lpips_metric(pred, gt).item()),
    )


def mean_or_nan(vals: List[float]) -> float:
    """
    Compute the mean of a list.

    Returns NaN for empty inputs so downstream aggregation can
    distinguish missing data from valid zero-valued metrics.
    """
    return float(np.mean(vals)) if len(vals) > 0 else float("nan")


def write_placeholder_avg_row(
    writer: csv.writer,
    header: List[str],
) -> None:
    """
    Write a temporary dataset-average row.

    The final dataset averages are only known after all scenes
    have been processed, so this row is patched later.
    """
    row = ["__DATASET_AVG__"] + [""] * (len(header) - 1)
    writer.writerow(row)


def rewrite_first_data_row(
    csv_path: Path,
    avg_row: List[object],
) -> None:
    """
    Replace the placeholder dataset-average row in a CSV file.

    The CSV is streamed scene-by-scene during evaluation and the
    dataset-average row is updated once all scene statistics are
    available.
    """
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")

    with csv_path.open("r", newline="") as rf:
        rows = list(csv.reader(rf))

    if len(rows) < 2:
        raise RuntimeError(
            f"CSV too short to patch avg row: {csv_path}"
        )

    rows[1] = [
        str(x) if x is not None else ""
        for x in avg_row
    ]

    with tmp.open("w", newline="") as wf:
        w = csv.writer(wf)

        for r in rows:
            w.writerow(r)

    os.replace(tmp, csv_path)
