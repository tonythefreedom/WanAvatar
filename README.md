# WanAvatar

Lip-sync video generation based on Wan2.2-S2V-14B model with StableAvatar-inspired frontend.

## Features

- **Wan2.2-S2V-14B Model**: High-quality 14B parameter Speech-to-Video model
- **Custom Safetensors Loading**: Optimized loading with CPU offload support for memory efficiency
- **Multi-language UI**: Support for English, Korean (한국어), and Chinese (中文)
- **Gradio Web Interface**: Easy-to-use web application
- **GPU Memory Optimization**: Multiple modes including Normal, CPU Offload, and Sequential CPU Offload
- **Audio Tools**: Audio extraction from video and vocal separation

## Requirements

- Python 3.10+
- CUDA 11.8+
- GPU with 13GB+ VRAM (with CPU offload) or 25GB+ VRAM (Normal mode)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/tonythefreedom/WanAvatar.git
cd WanAvatar
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements_wan22.txt
pip install -r requirements_s2v.txt

# Install flash-attn (optional but recommended)
pip install flash-attn==2.6.3 --no-build-isolation
```

### 4. Download model checkpoints

Download the Wan2.2-S2V-14B model and place it in the `checkpoints/` directory:

```
checkpoints/
├── Wan2.2-S2V-14B/
│   ├── config.json
│   ├── diffusion_pytorch_model-00001-of-00007.safetensors
│   ├── diffusion_pytorch_model-00002-of-00007.safetensors
│   ├── diffusion_pytorch_model-00003-of-00007.safetensors
│   ├── diffusion_pytorch_model-00004-of-00007.safetensors
│   ├── diffusion_pytorch_model-00005-of-00007.safetensors
│   ├── diffusion_pytorch_model-00006-of-00007.safetensors
│   ├── diffusion_pytorch_model-00007-of-00007.safetensors
│   ├── diffusion_pytorch_model.safetensors.index.json
│   └── vae/
├── xlm-roberta-large/
├── wav2vec2-large-xlsr-53/
└── Kim_Vocal_2.onnx
```

## Usage

### Gradio Web Interface

```bash
# Local access
python app.py

# With public tunneling (accessible from anywhere)
python app.py --share
```

The application will start at `http://0.0.0.0:7891`. With `--share`, a public Gradio URL will be generated.

### FastAPI Server + React Frontend

```bash
# Start the server
./start.sh

# Or manually
python server.py
```

Access points:
- Frontend: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

### GPU Memory Modes

| Mode | VRAM Usage | Description |
|------|------------|-------------|
| Normal | ~25GB | Full GPU inference |
| model_cpu_offload | ~13GB | Model stays on CPU, moves to GPU during inference |
| model_cpu_offload_and_qfloat8 | ~10GB | CPU offload with 8-bit quantization |
| sequential_cpu_offload | ~8GB | Sequential layer-by-layer CPU offload |

### Language Support

The UI supports three languages with automatic translation of prompts:

- **English**: Full English interface with English negative prompts
- **한국어**: Korean interface with Korean negative prompts
- **中文**: Chinese interface with Chinese negative prompts

## Project Structure

```
WanAvatar/
├── app.py                    # Gradio web application (fullscreen layout)
├── server.py                 # FastAPI server for REST API
├── start.sh                  # Server startup script
├── generate.py               # CLI generation script
├── inference.py              # Inference utilities
├── preprocess_training_data.py  # Data preprocessing for LoRA training
├── train_lora.sh             # LoRA training script (1.3B model)
├── train_lora_14B.sh         # LoRA training script (14B model)
├── frontend/                 # React frontend application
│   ├── src/
│   │   ├── App.jsx           # Main React component
│   │   ├── api.js            # API client
│   │   └── ...
│   ├── package.json
│   └── vite.config.js
├── wan/
│   ├── speech2video.py       # Main S2V pipeline with custom safetensors loading
│   ├── modules/
│   │   ├── s2v/              # Speech-to-Video model components
│   │   │   ├── model_s2v.py  # WanModel_S2V transformer
│   │   │   ├── audio_encoder.py
│   │   │   └── ...
│   │   └── vae2_1.py         # VAE encoder/decoder
│   ├── configs/              # Model configurations
│   ├── dataset/              # Dataset loaders for training
│   ├── utils/                # Utilities (solvers, optimizations, lora)
│   └── distributed/          # Distributed training utilities
├── examples/                 # Example images for testing
├── accelerate_config/        # Accelerate configuration files
├── deepspeed_config/         # DeepSpeed configuration files
├── requirements.txt          # Base dependencies
├── requirements_wan22.txt    # Wan2.2 specific dependencies
└── requirements_s2v.txt      # S2V specific dependencies
```

## API Endpoints (FastAPI Server)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Generate video from image + audio |
| `/api/extract-audio` | POST | Extract audio from video |
| `/api/separate-vocals` | POST | Separate vocals from audio |
| `/api/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation |

## Key Modifications

### Custom Safetensors Loading

The `wan/speech2video.py` includes custom safetensors loading functions to handle the 14B model:

```python
def load_safetensors(path, device="cpu"):
    """Load safetensors file to specified device."""
    tensors = {}
    with safe_open(path, framework="pt", device=device) as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def load_sharded_safetensors(checkpoint_dir):
    """Load sharded safetensors files from a directory."""
    # Supports both index file and glob pattern loading
    ...
```

This approach:
- Loads model weights to CPU first to avoid CUDA OOM
- Supports sharded safetensors files
- Moves to GPU only during inference (with offload mode)

## Person-Specific Lip-Sync Fine-Tuning (LoRA)

특정 인물에 대한 립싱크 품질을 크게 향상시키기 위해 LoRA(Low-Rank Adaptation) 파인튜닝을 수행할 수 있습니다.

### Overview

LoRA 파인튜닝은 기본 모델의 가중치를 동결하고, 저차원 행렬을 통해 적은 파라미터만 학습합니다:
- **학습 파라미터**: ~0.1% of full model
- **학습 시간**: 약 3-4시간 (A100 80GB 기준, 10 epochs)
- **VRAM 요구량**: ~45-55GB (DeepSpeed ZeRO Stage 2 + gradient checkpointing)
- **RAM 요구량**: ~100GB (모델 로딩 + CPU optimizer offload)

### 0. 추가 모델 다운로드

학습에는 CLIP 이미지 인코더 모델이 필요합니다. OpenCLIP에서 자동 다운로드:

```python
import open_clip
import torch

# CLIP 모델 다운로드 및 저장
model, _, _ = open_clip.create_model_and_transforms(
    'xlm-roberta-large-ViT-H-14',
    pretrained='frozen_laion5b_s13b_b90k'
)
visual_state_dict = model.visual.state_dict()
torch.save(
    visual_state_dict,
    '/path/to/Wan2.2-S2V-14B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
)
```

필요한 패키지:
```bash
pip install open-clip-torch
```

### 1. 데이터 준비

#### 1.1 필요한 데이터

| 항목 | 권장 사양 |
|------|----------|
| 영상 개수 | 10-50개 |
| 영상 길이 | 2-10초 |
| 해상도 | 512x512 이상 |
| 프레임레이트 | 25fps |
| 오디오 | 깨끗한 음성, 배경 노이즈 최소화 |
| 얼굴 | 정면 또는 약간의 측면, 입이 잘 보이는 앵글 |

#### 1.2 데이터 디렉토리 구조

각 학습 영상은 다음 구조로 전처리되어야 합니다:

```
talking_face_data/
├── video_001/
│   ├── sub_clip.mp4          # 원본 비디오 클립
│   ├── audio.wav             # 추출된 오디오 (16kHz)
│   ├── images/               # 프레임 이미지
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   └── ...
│   ├── face_masks/           # 얼굴 마스크
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   └── ...
│   └── lip_masks/            # 입술 마스크
│       ├── 0001.png
│       ├── 0002.png
│       └── ...
├── video_002/
│   └── ...
└── video_path.txt            # 비디오 경로 목록
```

#### 1.3 데이터 전처리 스크립트

```bash
# 1. 비디오에서 프레임 추출
ffmpeg -i input_video.mp4 -vf "fps=25" images/%04d.png

# 2. 오디오 추출 (16kHz mono)
ffmpeg -i input_video.mp4 -ar 16000 -ac 1 audio.wav

# 3. 얼굴/입술 마스크 생성 (MediaPipe 또는 face-alignment 사용)
python lip_mask_extractor.py --video_dir talking_face_data/video_001
```

#### 1.4 경로 파일 생성

`video_path.txt` 파일에 각 비디오 디렉토리의 절대 경로를 한 줄씩 기록:

```
/path/to/talking_face_data/video_001
/path/to/talking_face_data/video_002
/path/to/talking_face_data/video_003
```

### 2. 학습 설정

#### 2.1 주요 하이퍼파라미터

| 파라미터 | 권장값 | 설명 |
|---------|--------|------|
| `learning_rate` | 1e-4 | LoRA 학습률 |
| `num_train_epochs` | 10-50 | 에폭 수 (10 권장, 품질/시간 균형) |
| `train_batch_size` | 1 | 배치 크기 (VRAM 제한) |
| `gradient_accumulation_steps` | 8 | Gradient 누적 (effective batch = 8) |
| `video_sample_n_frames` | 81 | 클립당 프레임 수 (81=약2초@25fps) |
| `lora_rank` | 128 | LoRA rank (높을수록 표현력↑, 메모리↑) |
| `max_grad_norm` | 0.05 | Gradient clipping |
| `motion_sub_loss` | True | 모션 손실 활성화 (립싱크 개선) |

#### 2.2 학습 스크립트 (`train_lora_14B.sh`)

`train_lora_14B.sh` 파일을 환경에 맞게 수정:

```bash
#!/bin/bash
# LoRA Training Script for WanAvatar 14B Model

cd /path/to/WanAvatar

export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/path/to/WanAvatar:/path/to/Wan2.2:$PYTHONPATH"
export MODEL_NAME="/path/to/Wan2.2-S2V-14B"
export WAV2VEC_PATH="/path/to/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english"

# Training data paths
export TRAIN_DATA="ft_data/processed/video_path.txt"
export VALIDATION_REF="validation/reference.png"
export VALIDATION_AUDIO="validation/audio.wav"
export OUTPUT_DIR="output_lora_14B"

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
  --train_data_rec_dir=$TRAIN_DATA \
  --train_data_vec_dir=$TRAIN_DATA \
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
```

**주요 파라미터 설명:**
| 파라미터 | 설명 |
|---------|------|
| `--use_deepspeed` | DeepSpeed 메모리 최적화 활성화 |
| `--deepspeed_config_file` | DeepSpeed ZeRO Stage 2 설정 파일 |
| `--train_data_rec_dir` | 가로 비디오 학습 데이터 경로 |
| `--train_data_vec_dir` | 세로 비디오 학습 데이터 경로 |
| `--gradient_accumulation_steps` | Gradient 누적 횟수 (메모리 절약) |
| `--video_repeat` | 비디오 반복 횟수 (데이터 증강) |
| `--vae_mini_batch` | VAE 미니배치 크기 (메모리 절약) |
| `--report_to` | 학습 로그 기록 방식 (tensorboard) |

#### 2.3 DeepSpeed 설정 (`deepspeed_config/zero2_config.json`)

80GB VRAM에서 학습하기 위해 DeepSpeed ZeRO Stage 2 + CPU Optimizer Offload를 사용합니다:

```json
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": 8,
    "gradient_clipping": 0.05,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": 1
}
```

**메모리 최적화 팁:**
- `offload_optimizer`: Optimizer 상태를 CPU로 오프로드하여 VRAM 절약
- `gradient_checkpointing`: 중간 활성화값 재계산으로 메모리 절약
- `gradient_accumulation_steps`: 높을수록 메모리 효율적 (16 권장)

### 3. 학습 실행

#### 3.1 단일 GPU 학습

```bash
# Accelerate 설정
accelerate config

# 학습 시작
bash train_14B_lora.sh
```

#### 3.2 다중 GPU 학습 (DeepSpeed)

```bash
# DeepSpeed 설정 파일 사용
accelerate launch --config_file accelerate_config/accelerate_config_machine_14B_multiple.yaml \
  train_14B_lora.py [arguments...]
```

### 4. 학습 모니터링

#### 4.1 TensorBoard

```bash
tensorboard --logdir output_lora/logs
```

#### 4.2 체크포인트 구조

```
output_lora/
├── checkpoint-500/
│   ├── pytorch_lora_weights.safetensors  # LoRA 가중치
│   └── optimizer.bin
├── checkpoint-1000/
│   └── ...
└── logs/
    └── events.out.tfevents.*
```

### 5. LoRA 가중치 적용

#### 5.1 추론 시 LoRA 로드

```python
from wan.utils.lora_utils import load_lora_weights

# 기본 모델 로드 후 LoRA 적용
model = load_base_model()
load_lora_weights(model, "output_lora/checkpoint-1000/pytorch_lora_weights.safetensors")
```

#### 5.2 LoRA 가중치 병합 (선택사항)

LoRA 가중치를 기본 모델에 병합하여 추론 속도 향상:

```python
# 병합 후 저장
merged_model = merge_lora_weights(model, lora_path)
merged_model.save_pretrained("merged_model")
```

### 6. 학습 팁

#### 6.1 데이터 품질

- **조명**: 균일한 조명, 역광 피하기
- **배경**: 단순하고 정적인 배경 권장
- **표정**: 자연스러운 말하기 표정, 과장된 표정 피하기
- **오디오**: 배경 음악/노이즈 제거 (Vocal Separation 기능 활용)

#### 6.2 과적합 방지

```bash
# 드롭아웃 추가
--lora_dropout=0.1

# 조기 종료
--early_stopping_patience=5
```

#### 6.3 메모리 최적화

```bash
# Gradient checkpointing 필수
--gradient_checkpointing

# 8-bit Adam optimizer
--use_8bit_adam

# VAE 미니배치
--vae_mini_batch=1
```

### 7. 성능 비교

| 설정 | 립싱크 품질 | 학습 시간 | VRAM |
|------|------------|----------|------|
| Base Model | ★★★☆☆ | - | 13GB |
| + Parameter Optimization | ★★★★☆ | - | 13GB |
| + LoRA Fine-tuning (50 epochs) | ★★★★★ | ~2h | 40GB |

### 8. 문제 해결

#### CUDA Out of Memory

```bash
# gradient_checkpointing 활성화
--gradient_checkpointing

# 배치 크기 줄이기
--train_batch_size=1 --gradient_accumulation_steps=4

# 프레임 수 줄이기
--video_sample_n_frames=41  # 1초@25fps
```

#### 학습이 수렴하지 않는 경우

1. 학습률 조정: `1e-4` → `5e-5`
2. 데이터 품질 확인: 마스크가 정확한지 검증
3. 에폭 수 증가: 100+ epochs

## Credits

- **Wan2.2-S2V-14B**: Alibaba's Speech-to-Video model
- **StableAvatar**: Original frontend inspiration from [StableAvatar](https://github.com/Francis-Rings/StableAvatar)
- **Gradio**: Web interface framework

## License

This project is for research purposes. Please refer to the original model licenses for usage terms.
