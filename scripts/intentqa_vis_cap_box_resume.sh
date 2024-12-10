#!/bin/bash

#SBATCH -p gpu-he -C h100 --gres=gpu:2 --gres-flags=enforce-binding
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=128G
#SBATCH -t 48:00:00

nvidia-smi

# keep n_gpu x btz x accum_iter == 64

torchrun --rdzv_endpoint 127.0.0.1:1244 --nproc_per_node 2 train.py \
--num_workers 16 \
--model 8B \
--resume ./checkpoint/intentqa_cap_box_resume/checkpoint_best.pth \
--max_seq_len 2100 \
--batch_size 1 \
--epochs 10 \
--warmup_epochs 2 \
--bias 3.5 \
--tau 100. \
--dataset intentqa \
--blr 9e-2 \
--weight_decay 0.14 \
--accum_iter 32 \
--use_vis \
--use_cap \
--use_box \
--box_format textual \
--box_max_feats 1 \
--box_input_dim 4 \
--llama_model_path finetuned/pretrained/llama3/ \
--output_dir ./checkpoint/intentqa_vis_cap_box_resume \
--project_name intentqa \
--zero_init