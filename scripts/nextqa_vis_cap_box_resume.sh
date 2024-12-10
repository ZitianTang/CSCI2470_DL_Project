#!/bin/bash

#SBATCH -p cs-superlab-gcondo --gres=gpu:8 --gres-flags=enforce-binding
#SBATCH --account=cs-superlab-gcondo
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=1000G
#SBATCH -t 24:00:00

nvidia-smi

# keep n_gpu x btz x accum_iter == 64

torchrun --rdzv_endpoint 127.0.0.1:1242 --nproc_per_node 8 train.py \
--num_workers 16 \
--model 8B \
--resume ./checkpoint/nextqa_cap_box_textual_resume/checkpoint_best.pth \
--max_seq_len 2100 \
--batch_size 1 \
--epochs 5 \
--warmup_epochs 1 \
--bias 3.5 \
--tau 100. \
--dataset nextqa \
--blr 9e-2 \
--weight_decay 0.14 \
--accum_iter 8 \
--use_vis \
--use_cap \
--use_box \
--box_format textual \
--box_max_feats 1 \
--box_input_dim 4 \
--llama_model_path finetuned/pretrained/llama3/ \
--output_dir ./checkpoint/nextqa_vis_cap_box_textual_resume \
--project_name nextqa \
--zero_init