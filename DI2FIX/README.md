---
<div align="center">

# DF3DV-1K: A Large-Scale Dataset and Benchmark for Distractor-Free Novel View Synthesis

[Cheng-You Lu](https://johnnylu305.github.io/)<sup>1</sup>, Yi-Shan Hung<sup>2</sup>, Wei-Ling Chi<sup>3*</sup>, Hao-Ping Wang<sup>3*</sup>,  Charlie Li-Ting Tsai<sup>1*</sup>, Yu-Cheng Chang<sup>1</sup>, [Yu-Lun Liu](https://yulunalexliu.github.io/)<sup>3</sup>, Thomas Do<sup>1</sup>, [Chin-Teng Lin](https://scholar.google.com/citations?user=nubkF1cAAAAJ&hl=zh-TW)<sup>1</sup>

<p><sup>1</sup>University of Technology Sydney &nbsp;&nbsp;<sup>2</sup>University of Sydney&nbsp;&nbsp;<sup>3</sup>National Yang Ming Chiao Tung University&nbsp;&nbsp;
<br><sup>*</sup>Equal Contribution&nbsp;&nbsp;

### [Project Page](https://johnnylu305.github.io/df3dv1k_web/) · [Paper](https://arxiv.org/abs/2604.13416) · [Demo Page](https://chengyou305-di2fix-demo.hf.space/)

---

</div>

## Introduction
This codebase contains the official implementation of **DI<sup>2</sup>FIX** (Distractor-Free DIFIX), a plug-and-play 2D enhancer for improving radiance field renderings, introduced in the DF3DV-1K paper.

## Tested Environment
The codebase has been tested under the following environment:

| Component | Version |
|-----------|---------|
| Operating System | Red Hat Enterprise Linux 8.10 |
| GPU | 2 × NVIDIA L40S (48 GB) |
| CUDA | 12.8 |
| Python | 3.10 |

## Setup
```
git clone https://github.com/johnnylu305/DI2FIX.git
cd DI2FIX
pip install -r requirements.txt
```

## Directory Structure
<details>
<summary>Recommended directory structure</summary>

```
├── DF3DV-1K-Fixer (training and validation datasets)
│   ├── Train
│   │   └── DF3DV-1K-Star-Fixer
│   └── Val
│       ├── DF3DV-41-Fixer
│       └── on-the-go
└── DI2FIX (DI²FIX codebase)
    ├── src (source code)
    ├── outputs (checkpoints and training-time results)
    ├── results (inference-time results)
    ├── train_di2fix.sh (training script)
    ├── predict_di2fix.sh (inference script)
    └── eval.sh (evaluation script)
```
</details>
  
## Download Dataset
Please see the dataset download information and instructions [here](https://github.com/johnnylu305/DF3DV/tree/main#df3dv-1k-fixer).

## Reproduced Checkpoints and Results

The reproduced checkpoints and results can be downloaded [here](https://drive.google.com/drive/folders/12tytVVJ4UTsRh9jjtuKczNsEJwuxof_Z?usp=sharing) and [here](https://drive.google.com/file/d/1ckX7W40vrPX_Z7YB3sihM1P_g7va6VRE/view?usp=sharing), respectively.

PSNR:
| Method | 3DGS | T&#x2011;3DGS | T&#x2011;3DGS&#x2011;TMR | WildGaussian | SLS | DeSplat | DeGauss | OCSplats | RobustSplat | AsymGS | Overall |
|----------|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| Vanilla | 18.71 | 20.90 | 19.12 | 20.93 | 21.05 | 21.02 | 21.77 | 21.41 | 21.52 | 21.75 | 20.82 |
| DI<sup>2</sup>FIX | **+1.83** | **+1.26** | **+2.02** | **+0.77** | **+0.67** | **+0.86** | **+0.47** | **+0.63** | **+0.69** | **+0.39** | **+0.96** |

## Training
Run the default training script:
```
bash train_di2fix.sh
```
<details>
<summary>Example training command for 2+ GPUs</summary>

```
export NUM_NODES=1
export NUM_GPUS=2
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
```
</details>
  
<details>
<summary>Example training command for 1 GPU</summary>

```
accelerate launch --mixed_precision=no src/train_di2fix.py \
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
```
</details>

## Inference

Run the default inference script:
```
bash predict_di2fix.sh 
```
<details>
<summary>Example inference command</summary>

```
python3 src/predict_di2fix.py \
    --dataset_path ../DF3DV-1K-Fixer/dataset_df3dv1k.json \
    --ckpt outputs/df3dv1k/di2fix/train/checkpoints/model_87501.pkl \
    --out_dir results/di2fix/ \
    --split test \
    --save_grid \
    --lora_rank_vae 4 \
    --timestep 199 \
    --mixed_precision no \
    --mv_unet \
    --all_ids
```
</details>

## Evaluation

Run the default evaluation script:
```
bash eval.sh
```

<details>
<summary>Example evaluation command</summary>

```
python src/eval.py --root results/di2fix/grid/ \
    --mode fixed \
    --device cuda \
    --print_per_image
```
</details>

## Custom Dataset
If you would like to process your own data for training or evaluation, you can follow the preprocessing pipelines used for [DF3DV-1K](https://github.com/johnnylu305/DF3DV/tree/main/DF3DV_Data_Preparation) and [DF3DV-1K-Fixer](https://github.com/johnnylu305/DF3DV-1K/tree/main/Fixer_Data_Preparation).

## Acknowledgements
DI<sup>2</sup>FIX is built upon the following project:
- [Difix3D](https://github.com/nv-tlabs/Difix3D)
- [DF3DV-1K](https://github.com/johnnylu305/DF3DV-1K)
- [DF3DV-1K-Fixer](https://github.com/johnnylu305/DF3DV/tree/main#df3dv-1k-fixer)

## License
The license of DI<sup>2</sup>FIX follows the Difix3D [license](https://github.com/nv-tlabs/Difix3D/blob/main/LICENSE.txt).

## Citation
```
@article{lu2026df3dv,
  title={DF3DV-1K: A Large-Scale Dataset and Benchmark for Distractor-Free Novel View Synthesis},
  author={Lu, Cheng-You and Hung, Yi-Shan and Chi, Wei-Ling and Wang, Hao-Ping and Tsai, Charlie Li-Ting and Chang, Yu-Cheng and Liu, Yu-Lun and Do, Thomas and Lin, Chin-Teng},
  journal={arXiv preprint arXiv:2604.13416},
  year={2026}
}
```
