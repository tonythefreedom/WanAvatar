#!/bin/bash
# LoRA Training Script for WanAvatar 14B Model (Wan2.2-S2V-14B)

cd /home/work/WanAvatar

export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/work/WanAvatar:/home/work/musicfm:/home/work/Wan2.2:$PYTHONPATH"
export MODEL_NAME="/home/work/Wan2.2/Wan2.2-S2V-14B"
export WAV2VEC_PATH="/home/work/Wan2.2/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english"

# Training data paths
export TRAIN_DATA="/home/work/WanAvatar/ft_data/processed/video_path.txt"
export VALIDATION_REF="/home/work/WanAvatar/validation/reference.png"
export VALIDATION_AUDIO="/home/work/WanAvatar/validation/audio.wav"
export OUTPUT_DIR="/home/work/WanAvatar/output_lora_14B"

# Create output directory
mkdir -p $OUTPUT_DIR

accelerate launch \
  --mixed_precision=bf16 \
  --num_processes=1 \
  --use_deepspeed \
  --deepspeed_config_file="deepspeed_config/zero2_config.json" \
  train_14B_lora.py \
  --config_path="deepspeed_config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_wav2vec_path=$WAV2VEC_PATH \
  --validation_reference_path=$VALIDATION_REF \
  --validation_driven_audio_path=$VALIDATION_AUDIO \
  --train_data_rec_dir="/home/work/WanAvatar/ft_data/processed/video_path.txt" \
  --train_data_square_dir="/home/work/WanAvatar/ft_data/processed/video_path.txt" \
  --train_data_vec_dir="/home/work/WanAvatar/ft_data/processed/video_path.txt" \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=8 \
  --dataloader_num_workers=0 \
  --num_train_epochs=10 \
  --checkpointing_steps=100 \
  --validation_steps=10000 \
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
