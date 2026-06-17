export NUM_NODES=1
export NUM_GPUS=2
# Notes:
# - max_train_steps does not need to cover all images because many samples are similar.
# - Increase train_batch_size if GPU memory allows.
# - Increase checkpoint/eval/viz intervals if storage space is limited.
# - lambda_gram=0 disables Gram loss.
# - lambda_ssim 1.0 may improve SSIM if enabled.
accelerate launch --mixed_precision=no --main_process_port 29501 --num_machines $NUM_NODES --multi_gpu --num_processes $NUM_GPUS src/train_di2fix.py \
    --output_dir=./outputs/df3dv1k/di2fix/train \
    --dataset_path="../DF3DV-1K-Fixer/dataset_df3dv1k.json" \
    --max_train_steps 100000 \
    --learning_rate 2e-5 \
    --train_batch_size=1 --dataloader_num_workers 8 \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=2500 --eval_freq 2500 --viz_freq 500 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 0 --gram_loss_warmup_steps 2000 \
    --report_to "wandb" --tracker_project_name "di2fix" --tracker_run_name "di2fix-df3dv1k" --timestep 199 \
    --mv_unet
