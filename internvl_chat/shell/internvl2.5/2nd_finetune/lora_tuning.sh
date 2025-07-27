#!/bin/bash
set -eux  # Exit on error and show commands

# Install all required dependencies
pip install -U deepspeed timm transformers accelerate bitsandbytes peft flash-attn

# ---- Configuration ----
GPUS=1
BATCH_SIZE=4
PER_DEVICE_BATCH_SIZE=4
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

# Environment setup
export PYTHONPATH=$(pwd)/InternVL_for_psl/internvl_chat
export INTERNVL_DISABLE_FLASH_ATTN=1
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3

# Output directory
OUTPUT_DIR='/content/drive/MyDrive/psl_lora_output'
mkdir -p "$OUTPUT_DIR"

TRAIN_SCRIPT="/content/InternVL_for_psl/internvl_chat/internvl/train/internvl_chat_finetune.py"

# Combine all arguments into a single command
FULL_COMMAND="torchrun --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=${GPUS} --master_port=${MASTER_PORT} \
'$TRAIN_SCRIPT' \
--model_name_or_path 'OpenGVLab/InternVL2_5-1B' \
--conv_style 'internvl2_5' \
--use_fast_tokenizer False \
--output_dir '$OUTPUT_DIR' \
--meta_path '/content/drive/MyDrive/psl_dataset_train.json' \
--overwrite_output_dir \
--force_image_size 448 \
--max_dynamic_patch 6 \
--down_sample_ratio 0.5 \
--drop_path_rate 0.0 \
--freeze_llm \
--freeze_mlp \
--freeze_backbone \
--use_llm_lora 16 \
--vision_select_layer -1 \
--dataloader_num_workers 2 \
--bf16 \
--num_train_epochs 1 \
--per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
--gradient_accumulation_steps ${GRADIENT_ACC} \
--evaluation_strategy 'no' \
--save_strategy 'steps' \
--save_steps 200 \
--save_total_limit 1 \
--learning_rate 4e-5 \
--weight_decay 0.01 \
--warmup_ratio 0.03 \
--lr_scheduler_type 'cosine' \
--logging_steps 1 \
--max_seq_length 8192 \
--do_train \
--grad_checkpoint \
--group_by_length \
--dynamic_image_size \
--use_thumbnail \
--ps_version 'v2'"

# Execute the full command
echo "Executing command:"
echo "$FULL_COMMAND"
eval "$FULL_COMMAND" 2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
