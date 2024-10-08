#!/bin/bash
filename=$(basename "$0");exp_id="${filename%.*}"
CUDA_VISIBLE_DEVICES=1 \
python main.py \
--exp_id "$exp_id" --about "30w" \
--mode test \
--train_path ./dataset/hom_8u_16n_500_600/train \
--val_path ./dataset/hom_8u_16n_500_600/val \
--test_path  ./dataset/hom_8u_16n_500_600/test_8u_16n \
--user_num 8 \
--antenna_num 16 \
--batch_size 64 \
--end_epoch 1000 \
--model GCN \
--loss_type sum \
--edge_dim 2 \
--p_max 1 \
--save_model true --progress_bar false \
--lr 2e-4  --patience 2 --factor 0.1 --weight_decay 0 \
