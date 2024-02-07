#!/bin/bash
filename=$(basename "$0");exp_id="${filename%.*}"
CUDA_VISIBLE_DEVICES=0 \
python main.py \
--exp_id "$exp_id" --about "30w" \
--mode test \
--train_path ./dataset/8u_16n_40w/train \
--val_path ./dataset/8u_16n_40w/val \
--test_path  ./dataset/8u_16n_40w/2p_test_7u \
--user_num 7 \
--antenna_num 16 \
--batch_size 128 \
--end_epoch 1000 \
--model EdgeComGATBF \
--type BF \
--dataset_type Com \
--edge_dim 3 \
--p_max 2 \
--save_model true --progress_bar false \
--lr 5e-4  --patience 2 --factor 0.5 --weight_decay 0 \
