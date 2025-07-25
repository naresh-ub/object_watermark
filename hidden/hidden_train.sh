#!/bin/bash

# List of bit values to iterate through
NUM_BITS_LIST=(4 8 10 12 20)
GPU=7

# Set a different master port to avoid conflict
MASTER_PORT=29507
TRAIN_DIR=/home/csgrad/devulapa/watermark_final/data/finetune_ldm_train
VAL_DIR=/home/csgrad/devulapa/watermark_final/data/finetune_ldm_val

# Loop through each bit value
for NUM_BITS in "${NUM_BITS_LIST[@]}"; do
  PRETRAIN_DIR="output_${NUM_BITS}"
  OUTPUT_DIR="output_${NUM_BITS}_new"
  

  echo "Running training with NUM_BITS=${NUM_BITS}..."

  CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main.py \
    --val_dir ${VAL_DIR} \
    --train_dir ${TRAIN_DIR} \
    --output_dir ${OUTPUT_DIR} --eval_freq 5 \
    --img_size 256 --num_bits ${NUM_BITS} --batch_size 16 --epochs 300 \
    --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5 \
    --optimizer Lamb,lr=2e-2 \
    --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
    --scaling_w 0.3 --scale_channels False --attenuation none \
    --loss_w_type bce --loss_margin 12 \
    --resume_from ${PRETRAIN_DIR}/checkpoint180.pth

done