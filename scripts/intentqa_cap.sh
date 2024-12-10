#!/bin/bash

#SBATCH -p gpu-he -C a6000 --gres=gpu:1 --gres-flags=enforce-binding
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 48:00:00

nvidia-smi

# keep n_gpu x btz x accum_iter == 64

torchrun --rdzv_endpoint 127.0.0.1:1244 --nproc_per_node 1 train.py \
--model 8B \
--max_seq_len 1100 \
--batch_size 1 \
--epochs 10 \
--warmup_epochs 2 \
--bias 3.5 \
--tau 100. \
--dataset intentqa \
--blr 9e-2 \
--weight_decay 0.14 \
--accum_iter 64 \
--use_cap \
--llama_model_path finetuned/pretrained/llama3/ \
--output_dir ./checkpoint/intentqa_cap \
--project_name intentqa \
--zero_init