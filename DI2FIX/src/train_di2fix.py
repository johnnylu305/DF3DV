import os
import gc
import lpips
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
import transformers
from torchvision.transforms.functional import crop
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from glob import glob
from einops import rearrange

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb


from model import Di2fix, load_ckpt_from_state_dict, save_ckpt, Di2fix
from dataset import PairedDataset
from loss import gram_loss

import imageio
from datetime import timedelta
from pipeline_difix import DifixPipeline
import dataset_info
import time
from di2fix_utils import make_vis_grid, get_psnr, compute_metrics, to_uint8, load_pipe_weights_into_model
from torchmetrics.image import StructuralSimilarityIndexMeasure


def evaluate(accelerator, net_di2fix, dl, args, weight_dtype, net_lpips, save_step=0):
    """
    Run validation using the current model checkpoint.

    Metrics are aggregated per scene, per dataset, and overall.
    Qualitative results are optionally saved and logged to W&B.
    """

    def get_pred_fn(batch_val, x_src, x_tgt):
        return accelerator.unwrap_model(net_di2fix)(
            x_src,
            prompt_tokens=batch_val["input_ids"].to(accelerator.device),
        )

    return evaluate_core(
        accelerator,
        dl,
        args,
        weight_dtype,
        net_lpips,
        get_pred_fn,
        log_prefix="val",
        save_step=save_step,
    )


def evaluate_core(
    accelerator,
    dl,
    args,
    weight_dtype,
    net_lpips,
    get_pred_fn,
    *,
    log_prefix="val",
    save=True,
    save_step=0,
):
    """
    Run validation and compute PSNR and LPIPS.

    Aggregation hierarchy:

        (1) Dataset-method average:
                For each dataset and method, first compute the mean metric
                for each scene, then average those scene-level scores.
                -> logs["val/<dataset>/<metric>/<method>"]

        (2) Dataset average:
                For each dataset, average over all methods from (1).
                -> logs["val/<dataset>/<metric>"]

        (3) Overall average:
                Average over all datasets from (2).
                -> logs["val/<metric>"]

    Scene-level scores are used only for fair aggregation and are not logged.
    """
    methods = dataset_info.METHODS
    eval_ids = set(dataset_info.EVAL_IDS)
    dataset_ids = dataset_info.DATASET_IDS
    wandb_ids = set(dataset_info.WANDB_IDS)

    metrics_list = ["psnr", "lpips"]

    logs = {}
    logs["sample/results/SLS"] = []

    # scene -> method -> metric -> list[float]
    per_scene_method = {}

    for step, batch_val in enumerate(dl):
        method = batch_val["method"][0]
        scene_id = batch_val["scene_id"][0]
        series = batch_val["series"][0]
        eval_id = f"{scene_id}_{series}"

        if eval_id not in eval_ids:
            continue

        x_src = batch_val["conditioning_pixel_values"].to(
            accelerator.device, dtype=weight_dtype
        )
        x_tgt = batch_val["output_pixel_values"].to(
            accelerator.device, dtype=weight_dtype
        )

        with torch.no_grad():
            x_pred = get_pred_fn(batch_val, x_src, x_tgt)

            # Remove padding before visualization and metric computation.
            org_h = int(batch_val["org_h"][0])
            org_w = int(batch_val["org_w"][0])
            x_src = x_src[:, :, :, :org_h, :org_w]
            x_tgt = x_tgt[:, :, :, :org_h, :org_w]
            x_pred = x_pred[:, :, :, :org_h, :org_w]

            # Save qualitative results only for SLS to avoid excessive logging.
            if save and method == "SLS":
                grid = make_vis_grid(x_src, x_pred, x_tgt)
                # online
                if eval_id in wandb_ids:
                    logs["sample/results/SLS"].append(
                        wandb.Image(
                            to_uint8(
                                rearrange(grid, "b v c h w -> b c (v h) w")[0].cpu()
                            )
                        )
                    )
                # local
                save_dir = os.path.join(wandb.run.dir, "test", method, scene_id)
                os.makedirs(save_dir, exist_ok=True)

                img = rearrange(grid, "b v c h w -> b (v h) w c")[0].detach().cpu()
                img = to_uint8(img).numpy()

                filename = os.path.join(save_dir, f"{series}_{save_step:08d}.png")
                imageio.imwrite(filename, img)

            # Compute metrics on the first/source view only.
            x_pred_1 = x_pred[:, 0]
            x_tgt_1 = x_tgt[:, 0]
            metrics = compute_metrics(x_pred_1, x_tgt_1, net_lpips)

        per_scene_method.setdefault(scene_id, {})
        per_scene_method[scene_id].setdefault(
            method, {metric: [] for metric in metrics_list}
        )

        for metric in metrics_list:
            per_scene_method[scene_id][method][metric].append(float(metrics[metric]))

    # Convert image-level metrics into scene-level averages.
    scene_metric = {}
    for scene_id, method_dict in per_scene_method.items():
        scene_metric.setdefault(scene_id, {})

        for method, metric_dict in method_dict.items():
            scene_metric[scene_id].setdefault(method, {})

            for metric in metrics_list:
                vals = metric_dict[metric]
                if len(vals) == 0:
                    continue

                scene_metric[scene_id][method][metric] = float(np.mean(vals))

    # (1) Dataset-method average:
    # For each dataset and method, average over scene-level scores.
    dataset_method_metric = {}

    for dataset_name, scene_list in dataset_ids.items():
        dataset_method_metric.setdefault(dataset_name, {})

        for method in methods:
            dataset_method_metric[dataset_name].setdefault(method, {})

            for metric in metrics_list:
                vals = []

                for scene_id in scene_list:
                    if (
                        scene_id in scene_metric
                        and method in scene_metric[scene_id]
                        and metric in scene_metric[scene_id][method]
                    ):
                        vals.append(scene_metric[scene_id][method][metric])

                if len(vals) == 0:
                    continue

                value = float(np.mean(vals))
                dataset_method_metric[dataset_name][method][metric] = value
                logs[f"{log_prefix}_methods/{dataset_name}/{metric}/{method}"] = value

    # (2) Dataset average:
    # For each dataset and metric, average over all method-level scores.
    dataset_metric = {}

    for dataset_name in dataset_ids:
        dataset_metric.setdefault(dataset_name, {})

        for metric in metrics_list:
            vals = []

            for method in methods:
                if (
                    dataset_name in dataset_method_metric
                    and method in dataset_method_metric[dataset_name]
                    and metric in dataset_method_metric[dataset_name][method]
                ):
                    vals.append(dataset_method_metric[dataset_name][method][metric])

            if len(vals) == 0:
                continue

            value = float(np.mean(vals))
            dataset_metric[dataset_name][metric] = value
            logs[f"{log_prefix}_datasets/{dataset_name}/{metric}"] = value

    # (3) Overall average:
    # For each metric, average over dataset-level scores.
    for metric in metrics_list:
        vals = []

        for dataset_name in dataset_ids:
            if dataset_name in dataset_metric and metric in dataset_metric[dataset_name]:
                vals.append(dataset_metric[dataset_name][metric])

        if len(vals) == 0:
            continue

        logs[f"{log_prefix}_overall/{metric}"] = float(np.mean(vals))

    return logs


def main(args):
    # longer timeout
    ddp_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3000))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    net_di2fix = Di2fix(
        lora_rank_vae=args.lora_rank_vae, 
        timestep=args.timestep,
        mv_unet=args.mv_unet,
    )

    # Original difix
    pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
    pipe.to("cuda")
    load_pipe_weights_into_model(pipe, net_di2fix, report=True)
    del pipe

    net_di2fix.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_di2fix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_di2fix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_lpips = lpips.LPIPS(net='vgg').cuda()

    net_lpips.requires_grad_(False)
    
    net_vgg = torchvision.models.vgg16(pretrained=True).features
    for param in net_vgg.parameters():
        param.requires_grad_(False)

    net_ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to("cuda")
    net_ssim.requires_grad_(False)

    # make the optimizer
    layers_to_opt = []
    layers_to_opt += list(net_di2fix.unet.parameters())
   
    for n, _p in net_di2fix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt = layers_to_opt + list(net_di2fix.vae.decoder.skip_conv_1.parameters()) + \
        list(net_di2fix.vae.decoder.skip_conv_2.parameters()) + \
        list(net_di2fix.vae.decoder.skip_conv_3.parameters()) + \
        list(net_di2fix.vae.decoder.skip_conv_4.parameters())

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    dataset_train = PairedDataset(dataset_path=args.dataset_path, split="train", tokenizer=net_di2fix.tokenizer)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dataset_val = PairedDataset(dataset_path=args.dataset_path, split="test", tokenizer=net_di2fix.tokenizer)
    random.Random(42).shuffle(dataset_val.img_ids)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=args.dataloader_num_workers)


    # Resume from checkpoint
    global_step = 0    
    if args.resume is not None:
        if os.path.isdir(args.resume):
            # Resume from last ckpt
            ckpt_files = glob(os.path.join(args.resume, "*.pkl"))
            assert len(ckpt_files) > 0, f"No checkpoint files found: {args.resume}"
            ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split("/")[-1].replace("model_", "").replace(".pkl", "")))
            print("="*50); print(f"Loading checkpoint from {ckpt_files[-1]}"); print("="*50)
            global_step = int(ckpt_files[-1].split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_di2fix, optimizer = load_ckpt_from_state_dict(
                net_di2fix, optimizer, ckpt_files[-1]
            )
        elif args.resume.endswith(".pkl"):
            print("="*50); print(f"Loading checkpoint from {args.resume}"); print("="*50)
            global_step = int(args.resume.split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_di2fix, optimizer = load_ckpt_from_state_dict(
                net_di2fix, optimizer, args.resume
            )    
        else:
            raise NotImplementedError(f"Invalid resume path: {args.resume}")
    else:
        print("="*50); print(f"Training from scratch"); print("="*50)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_di2fix.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_vgg.to(accelerator.device, dtype=weight_dtype)
    net_ssim.to(accelerator.device, dtype=weight_dtype)    

    # Prepare everything with our `accelerator`.
    net_di2fix, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_di2fix, optimizer, dl_train, lr_scheduler
    )
    net_lpips, net_vgg, net_ssim = accelerator.prepare(net_lpips, net_vgg, net_ssim)    
    # renorm with image net statistics
    t_vgg_renorm =  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        init_kwargs = {
            "wandb": {
                "name": args.tracker_run_name,
                "dir": args.output_dir,
            },
        }        
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs=init_kwargs)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # compute validation set L2, LPIPS
    if accelerator.is_main_process:
        start = time.time()
        logs = evaluate(accelerator, net_di2fix, dl_val, args, weight_dtype, net_lpips, save_step=1)
        end = time.time()
        #print("Eval Time", end-start)
        gc.collect()
        torch.cuda.empty_cache()
        accelerator.log(logs, step=0)

    # start the training loop
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_di2fix]
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                B, V, C, H, W = x_src.shape

                # [B, 1, 1, H, W]
                nopad_mask = batch["nopad_mask"]

                # forward pass
                x_tgt_pred = net_di2fix(x_src, prompt_tokens=batch["input_ids"])       

                # apply mask
                x_tgt = x_tgt * nopad_mask
                x_tgt_pred = x_tgt_pred * nopad_mask

                x_tgt = rearrange(x_tgt, 'b v c h w -> (b v) c h w')
                x_tgt_pred = rearrange(x_tgt_pred, 'b v c h w -> (b v) c h w')
                         
                # Reconstruction loss
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips

                # DSSIM loss (optional)
                if args.lambda_ssim > 0:
                    # torchmetrics SSIM expects (N,C,H,W)
                    ssim_val = net_ssim(x_tgt_pred.float(), x_tgt.float())  # scalar
                    # dssim
                    loss_dssim = (1.0 - ssim_val) * 0.5
                    loss = loss + loss_dssim * args.lambda_ssim
                else:
                    loss_dssim = torch.tensor(0.0, device=x_tgt_pred.device)

                # Gram matrix loss
                if args.lambda_gram > 0:
                    if global_step > args.gram_loss_warmup_steps:
                        x_tgt_pred_renorm = t_vgg_renorm(x_tgt_pred * 0.5 + 0.5)
                        crop_h, crop_w = 400, 400
                        top, left = random.randint(0, H - crop_h), random.randint(0, W - crop_w)
                        x_tgt_pred_renorm = crop(x_tgt_pred_renorm, top, left, crop_h, crop_w)
                        
                        x_tgt_renorm = t_vgg_renorm(x_tgt * 0.5 + 0.5)
                        x_tgt_renorm = crop(x_tgt_renorm, top, left, crop_h, crop_w)
                        
                        loss_gram = gram_loss(x_tgt_pred_renorm.to(weight_dtype), x_tgt_renorm.to(weight_dtype), net_vgg) * args.lambda_gram
                        loss += loss_gram
                    else:
                        loss_gram = torch.tensor(0.0).to(weight_dtype)                    

                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                
                x_tgt = rearrange(x_tgt, '(b v) c h w -> b v c h w', v=V)
                x_tgt_pred = rearrange(x_tgt_pred, '(b v) c h w -> b v c h w', v=V)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    if args.lambda_gram > 0:
                        logs["loss_gram"] = loss_gram.detach().item()
                    progress_bar.set_postfix(**logs)

                    # viz some images
                    if global_step % args.viz_freq == 1:
                        # save training results
                        #print("save train")
                        grid = make_vis_grid(x_src, x_tgt_pred, x_tgt)
                        # online
                        logs["train/results"] = [wandb.Image(to_uint8(rearrange(grid, "b v c h w -> b c (v h) w")[idx].float().detach().cpu()), 
                            caption=f"idx={idx}") for idx in range(B)]
                        accelerator.log(logs, step=global_step)
                        # local
                        save_dir = os.path.join(wandb.run.dir, "train", "images")
                        os.makedirs(save_dir, exist_ok=True)
                        for idx in range(B):
                            img = to_uint8(rearrange(grid, "b v c h w -> b (v h) w c")[idx].float().detach().cpu())
                            filename = os.path.join(save_dir, f"{global_step:08d}_{idx:02d}.png")
                            imageio.imwrite(filename, img)


                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        # accelerator.unwrap_model(net_di2fix).save_model(outf)
                        save_ckpt(accelerator.unwrap_model(net_di2fix), optimizer, outf)

                    # compute validation set L2, LPIPS
                    if args.eval_freq > 0 and global_step % args.eval_freq == 1 and global_step>1:
                        start = time.time()
                        logs = evaluate(accelerator, net_di2fix, dl_val, args, weight_dtype, net_lpips, save_step=global_step)
                        end = time.time()
                        #print("Eval Time", end-start)

                        gc.collect()
                        torch.cuda.empty_cache()
                    accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break        


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # args for the loss function
    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_gram", default=1.0, type=float)
    parser.add_argument("--lambda_ssim", default=0.0, type=float) # enable this may have better ssim
    parser.add_argument("--gram_loss_warmup_steps", default=2000, type=int)

    # dataset options
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--prompt", default=None, type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="di2fix", help="The name of the wandb project to log to.")
    parser.add_argument("--tracker_run_name", type=str, required=True)

    # details about the model architecture
    parser.add_argument("--pretrained_model_name_or_path")
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument("--timestep", default=199, type=int)
    parser.add_argument("--mv_unet", action="store_true")

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,) # not used
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=50) # 10
    parser.add_argument("--max_train_steps", type=int, default=10_000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)
    
    # resume
    parser.add_argument("--resume", default=None, type=str)

    args = parser.parse_args()

    main(args)
