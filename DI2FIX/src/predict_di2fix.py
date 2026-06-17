import os
import argparse
import random

import numpy as np
import torch
import imageio
from einops import rearrange
from tqdm.auto import tqdm

from model import Di2fix
from dataset import PairedDataset
from pipeline_difix import DifixPipeline
from di2fix_utils import make_vis_grid, to_uint8, load_pipe_weights_into_model, load_finetuned_ckpt_model_only


@torch.no_grad()
def run_prediction(
    model,
    dl,
    device,
    weight_dtype,
    out_dir,
    save_grid,
):
    fixed_result_path = os.path.join(out_dir, "fixed2D")
    os.makedirs(fixed_result_path, exist_ok=True)

    if save_grid:
        grid_result_path = os.path.join(out_dir, "grid")
        os.makedirs(grid_result_path, exist_ok=True)

    model.eval()
    saved = 0

    for batch in tqdm(dl, desc="Predict", dynamic_ncols=True):
        # metadata
        method = batch["method"][0]
        scene_id = batch["scene_id"][0]
        series = batch["series"][0]
        dataset_id = batch["dataset_id"][0]

        # input and target
        x_src = batch["conditioning_pixel_values"].to(device, dtype=weight_dtype)
        x_tgt = batch["output_pixel_values"].to(device, dtype=weight_dtype)
        input_ids = batch["input_ids"].to(device)

        # original image size before padding
        org_h = int(batch["org_h"][0])
        org_w = int(batch["org_w"][0])

        # run DI2FIX
        x_pred = model(x_src, prompt_tokens=input_ids)

        # remove padding
        x_src = x_src[:, :, :, :org_h, :org_w]
        x_tgt = x_tgt[:, :, :, :org_h, :org_w]
        x_pred = x_pred[:, :, :, :org_h, :org_w]

        # save fixed source-view result
        save_dir = os.path.join(
            fixed_result_path,
            method,
            dataset_id,
            scene_id,
            f"{scene_id}-All",
            "images",
        )
        os.makedirs(save_dir, exist_ok=True)

        fixed = x_pred[:, 0].detach().cpu()[0]  # (C,H,W)
        fixed_hwc = rearrange(fixed, "c h w -> h w c")
        fixed_u8 = to_uint8(fixed_hwc).numpy()

        fixed_fn = os.path.join(save_dir, f"{series}.png")
        imageio.imwrite(fixed_fn, fixed_u8)

        # optionally save stitched grid
        if save_grid:
            grid_save_dir = os.path.join(
                grid_result_path,
                method,
                dataset_id,
                scene_id,
                f"{scene_id}-All",
                "images",
            )
            os.makedirs(grid_save_dir, exist_ok=True)

            grid = make_vis_grid(x_src, x_pred, x_tgt).detach().cpu()[0]
            grid_hwc = rearrange(grid, "v c h w -> (v h) w c")
            grid_u8 = to_uint8(grid_hwc).numpy()

            grid_fn = os.path.join(grid_save_dir, f"{series}.png")
            imageio.imwrite(grid_fn, grid_u8)

        saved += 1

    return saved


def build_argparser():
    p = argparse.ArgumentParser("predict_di2fix")

    p.add_argument("--dataset_path", required=True, type=str)
    p.add_argument("--split", default="test", choices=["train", "test"])

    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--out_dir", required=True, type=str)

    p.add_argument("--mv_unet", action="store_true")
    p.add_argument("--lora_rank_vae", default=4, type=int)
    p.add_argument("--timestep", default=199, type=int)

    p.add_argument(
        "--save_grid",
        action="store_true",
        help="Also save stitched 2x3 grid PNG per sample.",
    )

    p.add_argument("--num_workers", default=8, type=int)
    p.add_argument("--mixed_precision", default="no", choices=["no", "bf16"])
    p.add_argument("--seed", default=42, type=int)

    p.add_argument(
        "--all_ids",
        action="store_true",
        help="Do NOT filter by dataset_info.EVAL_IDS.",
    )

    return p


def main():
    args = build_argparser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight_dtype = torch.float32
    if args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    model = Di2fix(
        lora_rank_vae=args.lora_rank_vae,
        timestep=args.timestep,
        mv_unet=args.mv_unet,
    )

    # Load base Difix weights from Hugging Face.
    print("Loading base Difix weights from nvidia/difix_ref ...")
    pipe = DifixPipeline.from_pretrained(
        "nvidia/difix_ref",
        trust_remote_code=True,
    )
    pipe.to(device)
    load_pipe_weights_into_model(pipe, model, report=True)
    del pipe

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Load finetuned DI2FIX checkpoint.
    print(f"Loading finetuned checkpoint: {args.ckpt}")
    load_finetuned_ckpt_model_only(model, args.ckpt)

    model.to(device=device, dtype=weight_dtype)

    ds = PairedDataset(
        dataset_path=args.dataset_path,
        split=args.split,
        tokenizer=model.tokenizer,
        select_scenes=not args.all_ids,
        auto_size=True,
    )

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    n = run_prediction(
        model=model,
        dl=dl,
        device=device,
        weight_dtype=weight_dtype,
        out_dir=args.out_dir,
        save_grid=args.save_grid,
    )

    print(f"Done. Saved {n} samples into: {args.out_dir}")


if __name__ == "__main__":
    main()
