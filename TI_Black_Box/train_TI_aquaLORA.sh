## Experiment configs
export TRAIN_FILE="w_TI_48_bit_aquaLORA.py"

## Hyper-parameters
export LAMBDA_I=1.0
export LAMBDA_W=100
export RESOLUTION=512
export TRAIN_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=1
export MAX_TRAIN_STEPS=500000
export LR_WARMUP_STEPS=0
export NUM_BITS=10
# export NUM_VECTORS=2
export LOSS_I="watson-vgg"
export LOSS_W="bce"
export TI_RESOLUTION=512
export HIDDEN_RESOLUTION=256

export DECODER_PATH="/home/csgrad/devulapa/watermark_final/TI_Black_Box/assets/pretained_latentwm.pth"
export TRAIN_DIR="/home/csgrad/devulapa/watermark_final/data/finetune_ldm_debug"
export VAL_DIR="/home/csgrad/devulapa/watermark_final/data/finetune_ldm_val"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export HIDDEN_WHITEN_DIR="/home/csgrad/devulapa/watermark_final/data/finetune_ldm_train"
export CAPTIONS="/home/csgrad/devulapa/watermark_final/data/train_captions.json"
export VAL_CAPTIONS="/home/csgrad/devulapa/watermark_final/data/val_captions.json"

## Misc
export LR_SCHEDULER="constant"
export LEARNABLE_PROPERTY="object"
export PLACEHOLDER_TOKEN="<cat-toy>"
export INIT_TOKEN="toy"
export REPORT_TO="wandb"
export WANDB_PROJECT="Oct_23_baseline"

# Pass t_start and t_end as arguments
t_start=$1
t_end=$2
port=$3
gpu1=$4
report_to=$5
num_vectors=$6
output_dir=$7
lr=$8
# gpu2=$5

# Update TI_LAMBDA (if applicable)
export TI_LAMBDA=$t_start  # Assuming TI_LAMBDA is set based on t_start

# Create run name and output directory using t_start and t_end
export WANDB_RUN_NAME="T_start_${t_start}_T_end_${t_end}_${output_dir}"
export OUTPUT_DIR="train_logs/${output_dir}"

# Run the training script
CUDA_VISIBLE_DEVICES=$gpu1 accelerate launch --num_processes=1 --main_process_port=$port $TRAIN_FILE \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_dir=$TRAIN_DIR \
  --learnable_property=$LEARNABLE_PROPERTY \
  --placeholder_token=$PLACEHOLDER_TOKEN \
  --initializer_token=$INIT_TOKEN \
  --ti_resolution=$TI_RESOLUTION \
  --hidden_resolution=$HIDDEN_RESOLUTION \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --val_dir=$VAL_DIR \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --learning_rate=$lr \
  --scale_lr \
  --lr_scheduler=$LR_SCHEDULER \
  --lr_warmup_steps=$LR_WARMUP_STEPS \
  --output_dir=$OUTPUT_DIR \
  --msg_decoder_path=$DECODER_PATH \
  --num_bits=$NUM_BITS \
  --loss_i=$LOSS_I \
  --loss_w=$LOSS_W \
  --lambda_i=$LAMBDA_I \
  --lambda_w=$LAMBDA_W \
  --hidden_whiten_dir=$HIDDEN_WHITEN_DIR \
  --num_vectors=$6 \
  --train_captions=$CAPTIONS \
  --val_captions=$VAL_CAPTIONS \
  --wandb_project=$WANDB_PROJECT \
  --wandb_run_name=$WANDB_RUN_NAME \
  --t_start=$t_start \
  --t_end=$t_end \
  --report_to=$5 \
  --aqualora_pretrain_path=$DECODER_PATH \
