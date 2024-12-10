#!/bin/bash

#SBATCH -p cs-superlab-gcondo --gres=gpu:8 --gres-flags=enforce-binding
#SBATCH --account=cs-superlab-gcondo
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=1000G
#SBATCH -t 12:00:00

nvidia-smi

# keep n_gpu x btz x accum_iter == 64

torchrun --rdzv_endpoint 127.0.0.1:1242 --nproc_per_node 8 train.py \
--num_workers 16 \
--model 8B \
--max_seq_len 1100 \
--batch_size 1 \
--epochs 5 \
--warmup_epochs 1 \
--bias 3.5 \
--tau 100. \
--dataset nextqa \
--blr 9e-2 \
--weight_decay 0.14 \
--accum_iter 8 \
--use_cap \
--llama_model_path finetuned/pretrained/llama3/ \
--output_dir ./checkpoint/nextqa_cap \
--project_name nextqa \
--zero_init