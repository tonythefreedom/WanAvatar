#!/bin/bash
# LoRA Training Script for WanAvatar 1.3B Model

export TOKENIZERS_PARALLELISM=false
export MODEL_NAME="/home/work/StableAvatar/checkpoints/Wan2.1-Fun-V1.1-1.3B-InP"
export TRANSFORMER_CKPT="/home/work/StableAvatar/checkpoints/StableAvatar-1.3B/transformer3d-rec-vec.pt"
export WAV2VEC_PATH="/home/work/StableAvatar/checkpoints/wav2vec2-base-960h"

# Training data paths
export TRAIN_DATA="/home/work/WanAvatar/ft_data/processed/video_path.txt"
export VALIDATION_REF="/home/work/WanAvatar/validation/reference.png"
export VALIDATION_AUDIO="/home/work/WanAvatar/validation/audio.wav"
export OUTPUT_DIR="/home/work/WanAvatar/output_lora"

# Create output directory
mkdir -p $OUTPUT_DIR

accelerate launch \
  --mixed_precision=bf16 \
  --num_processes=1 \
  train_1B_rec_vec_lora.py \
  --config_path="deepspeed_config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --transformer_path=$TRANSFORMER_CKPT \
  --pretrained_wav2vec_path=$WAV2VEC_PATH \
  --validation_reference_path=$VALIDATION_REF \
  --validation_driven_audio_path=$VALIDATION_AUDIO \
  --train_data_rec_dir=$TRAIN_DATA \
  --train_data_vec_dir=$TRAIN_DATA \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=2 \
  --dataloader_num_workers=0 \
  --num_train_epochs=50 \
  --checkpointing_steps=200 \
  --validation_steps=10000 \
  --resume_from_checkpoint="checkpoint-200" \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --motion_sub_loss \
  --low_vram \
  --train_mode="i2v" \
  --report_to="tensorboard"
