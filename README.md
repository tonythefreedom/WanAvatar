# WanAvatar

Wan2.2-S2V-14B 기반 립싱크 비디오 생성 프로젝트. 음성 오디오와 참조 이미지를 입력하면 자연스러운 립싱크 비디오를 생성합니다.

## 주요 기능

- **Wan2.2-S2V-14B 모델**: 14B 파라미터 Speech-to-Video 모델
- **듀얼 GPU 추론**: T5 인코더 (GPU 1, ~15GB) + DiT 트랜스포머 (GPU 0, ~39GB) 분리 로딩
- **자동 분할 생성**: 긴 오디오를 자동으로 ~5초 세그먼트로 분할, Auto-Regressive 방식으로 연결
- **속도 최적화**: UniPC 솔버 (15 steps) + infer_frames 80 + TeaCache (실험적)
- **FastAPI 서버**: REST API 기반 비디오 생성 서버
- **Gradio 웹 인터페이스**: 간단한 웹 UI
- **LoRA 파인튜닝**: 특정 인물에 대한 립싱크 품질 향상

## 시스템 요구사항

| 항목 | 요구사항 |
|------|----------|
| Python | 3.12+ |
| CUDA | 12.4+ |
| GPU | 2x NVIDIA A100 80GB (권장) |
| VRAM | GPU 0: ~39GB (DiT), GPU 1: ~15GB (T5) |
| RAM | 64GB+ |
| 디스크 | ~60GB (모델 체크포인트) |

> **단일 GPU 모드**: 1x A100 80GB에서도 실행 가능 (CPU offload 활용)

## 설치

### 1. 리포지토리 클론

```bash
git clone https://github.com/tonythefreedom/WanAvatar.git
cd WanAvatar
```

### 2. 가상환경 생성

```bash
python3.12 -m venv venv
source venv/bin/activate
```

### 3. 의존성 설치

```bash
# PyTorch (CUDA 12.4)
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 프로젝트 의존성
pip install -r requirements.txt
pip install -r requirements_wan22.txt
pip install -r requirements_s2v.txt

# flash-attn (필수 - S2V 모델이 flash_attention()을 직접 호출)
pip install wheel
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

> **주의**: `flash-attn 2.8.3`은 PyTorch 2.6.0과 ABI 호환성 문제가 있어 `2.7.4.post1` 버전을 사용합니다.

### 4. 시스템 패키지

```bash
# ffmpeg (비디오+오디오 병합에 필요)
sudo apt-get install ffmpeg
```

### 5. 모델 다운로드

```bash
pip install huggingface-hub
huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir /mnt/models/Wan2.2-S2V-14B
```

모델 구조:

```
/mnt/models/Wan2.2-S2V-14B/
├── config.json
├── diffusion_pytorch_model-00001-of-00004.safetensors
├── diffusion_pytorch_model-00002-of-00004.safetensors
├── diffusion_pytorch_model-00003-of-00004.safetensors
├── diffusion_pytorch_model-00004-of-00004.safetensors
├── diffusion_pytorch_model.safetensors.index.json
├── Wan2.1_VAE.pth                          # Wan2.1 VAE (공식 모델에 포함)
├── models_t5_umt5-xxl-enc-bf16.pth        # T5 텍스트 인코더
├── google/umt5-xxl/                        # T5 토크나이저
└── wav2vec2-large-xlsr-53-english/         # Wav2Vec 오디오 인코더
```

> **참고**: Wan2.2-S2V-14B 공식 모델은 `Wan2.1_VAE.pth` (Wan2.1 VAE 아키텍처)를 사용합니다. 이는 정상입니다.

## 실행

### FastAPI 서버 (권장)

```bash
source venv/bin/activate
python server.py
```

- 서버: `http://localhost:8000`
- API 문서: `http://localhost:8000/docs`
- 프론트엔드: `http://localhost:8000` (빌드된 React 앱)

### Gradio 웹 인터페이스

```bash
python app.py           # 로컬 접속
python app.py --share   # 공개 URL 생성
```

포트: `http://0.0.0.0:7891`

## API 사용법

### 비디오 생성

```bash
# 1. 참조 이미지 업로드
IMAGE=$(curl -s -F "file=@reference.png" http://localhost:8000/api/upload/image)
IMAGE_PATH=$(echo $IMAGE | python3 -c "import sys,json; print(json.load(sys.stdin)['path'])")

# 2. 오디오 업로드
AUDIO=$(curl -s -F "file=@audio.wav" http://localhost:8000/api/upload/audio)
AUDIO_PATH=$(echo $AUDIO | python3 -c "import sys,json; print(json.load(sys.stdin)['path'])")

# 3. 생성 요청
TASK=$(curl -s -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d "{
    \"image_path\": \"$IMAGE_PATH\",
    \"audio_path\": \"$AUDIO_PATH\",
    \"resolution\": \"480*832\",
    \"inference_steps\": 15,
    \"infer_frames\": 80,
    \"seed\": 42
  }")
TASK_ID=$(echo $TASK | python3 -c "import sys,json; print(json.load(sys.stdin)['task_id'])")

# 4. 상태 확인
curl -s http://localhost:8000/api/status/$TASK_ID | python3 -m json.tool
```

### API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/api/generate` | POST | 이미지 + 오디오로 비디오 생성 |
| `/api/status/{task_id}` | GET | 생성 진행 상황 확인 |
| `/api/upload/image` | POST | 참조 이미지 업로드 |
| `/api/upload/audio` | POST | 오디오 파일 업로드 |
| `/api/extract-audio` | POST | 비디오에서 오디오 추출 |
| `/api/separate-vocals` | POST | 보컬 분리 |
| `/api/health` | GET | 서버 상태 확인 |

### 생성 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `resolution` | `720*1280` | 출력 해상도 (`480*832` 권장 - 빠름) |
| `inference_steps` | `15` | 디노이징 스텝 수 (높을수록 품질↑, 속도↓) |
| `infer_frames` | `80` | 세그먼트당 프레임 수 (~5초@16fps) |
| `guidance_scale` | `4.5` | CFG 스케일 |
| `seed` | `-1` | 랜덤 시드 (-1: 무작위) |
| `use_teacache` | `true` | TeaCache 캐싱 활성화 (실험적) |
| `teacache_thresh` | `0.1` | TeaCache 임계값 |
| `offload_model` | `false` | 모델 CPU 오프로드 |

## 아키텍처

### 듀얼 GPU 추론

2개 이상의 GPU가 감지되면 자동으로 듀얼 GPU 모드로 실행됩니다:

```
GPU 0 (~39GB): DiT 14B 트랜스포머 + VAE + Audio Encoder
GPU 1 (~15GB): T5 텍스트 인코더 (umt5-xxl)
```

- 모델은 CPU에서 먼저 로드 후 각 GPU로 이동 (OOM 방지)
- T5 인코더의 출력은 GPU 간 자동 전송

### Auto-Regressive 세그먼트 생성

긴 오디오(예: 25초)는 자동으로 여러 세그먼트로 분할됩니다:

```
[세그먼트 1] → [세그먼트 2] → [세그먼트 3] → ... → [최종 비디오]
     ↓              ↓              ↓
  참조 이미지    이전 마지막    이전 마지막
  기반 생성     프레임 참조    프레임 참조
```

- `infer_frames=80` → 세그먼트당 ~5초 (80프레임 / 16fps)
- 25초 오디오 → ~6개 세그먼트
- 각 세그먼트는 이전 세그먼트의 마지막 프레임을 motion reference로 사용

### 속도 최적화

| 최적화 | 효과 | 비고 |
|--------|------|------|
| UniPC 솔버 | 40 → 15 steps | 품질 유지하면서 2.7x 속도 향상 |
| infer_frames 80 | 세그먼트 수 감소 | 40 → 80 프레임으로 세그먼트 절반 |
| TeaCache | 유사 스텝 캐싱 | 실험적 - 조정 중 |

### 비디오 저장

생성된 비디오는 `save_videos_grid`로 저장 후 `ffmpeg`로 오디오를 병합합니다:

```python
# 비디오 프레임 저장
save_videos_grid(video[None], video_path, rescale=True, fps=16)

# 오디오 병합
ffmpeg -y -i video.mp4 -i audio.wav -c:v copy -c:a aac -shortest output.mp4
```

## 프로젝트 구조

```
WanAvatar/
├── server.py                 # FastAPI 서버 (포트 8000)
├── app.py                    # Gradio 웹 앱 (포트 7891)
├── start.sh                  # 서버 시작 스크립트
├── generate.py               # CLI 생성 스크립트
├── train_14B_lora.py         # 14B LoRA 학습 스크립트
├── train_lora_14B.sh         # LoRA 학습 실행 스크립트
├── preprocess_training_data.py  # 학습 데이터 전처리
├── lip_mask_extractor.py     # 얼굴/입술 마스크 추출
├── audio_extractor.py        # 오디오 추출
├── vocal_seperator.py        # 보컬 분리
├── frontend/                 # React 프론트엔드
│   ├── src/
│   │   ├── App.jsx
│   │   └── api.js
│   └── vite.config.js
├── wan/
│   ├── speech2video.py       # S2V 추론 파이프라인 (핵심)
│   │                         #   - 듀얼 GPU 지원
│   │                         #   - TeaCache 통합
│   │                         #   - Auto-Regressive 생성
│   ├── animate.py            # 애니메이션 파이프라인
│   ├── modules/
│   │   ├── s2v/              # S2V 모델 컴포넌트
│   │   │   ├── model_s2v.py  # WanModel_S2V 트랜스포머
│   │   │   ├── audio_encoder.py
│   │   │   └── motioner.py
│   │   ├── model.py          # 기본 모델 정의
│   │   ├── attention.py      # Flash Attention 2
│   │   ├── t5.py             # T5 텍스트 인코더
│   │   ├── vae2_1.py         # Wan2.1 VAE (S2V-14B 사용)
│   │   └── vae2_2.py         # Wan2.2 VAE (TI2V-5B 등 사용)
│   ├── configs/              # 모델 설정
│   │   └── wan_s2v_14B.py    # S2V 14B 설정
│   ├── models/
│   │   └── cache_utils.py    # TeaCache 계수 및 유틸리티
│   ├── utils/                # 유틸리티 (솔버, LoRA 등)
│   └── distributed/          # 분산 학습 유틸리티
├── examples/                 # 테스트용 예제 이미지
│   └── case-tony/
│       ├── reference01.png   # 464x688 참조 이미지
│       ├── reference02.png   # 512x768 참조 이미지
│       └── sample.wav        # 25.7초 테스트 오디오
├── outputs/                  # 생성된 비디오 출력
├── uploads/                  # 업로드된 파일
├── deepspeed_config/         # DeepSpeed 설정
│   └── wan2.2/
│       └── wan_s2v_14b.yaml
├── accelerate_config/        # Accelerate 설정
├── requirements.txt          # 기본 의존성
├── requirements_wan22.txt    # Wan2.2 의존성
└── requirements_s2v.txt      # S2V 의존성
```

## LoRA 파인튜닝

특정 인물에 대한 립싱크 품질을 향상시키기 위해 LoRA 파인튜닝을 수행할 수 있습니다.

### 개요

| 항목 | 사양 |
|------|------|
| 학습 파라미터 | ~0.1% of 14B 모델 |
| 학습 시간 | ~3-4시간 (A100 80GB, 10 epochs) |
| VRAM | ~45-55GB (DeepSpeed ZeRO Stage 2) |
| RAM | ~100GB (CPU optimizer offload) |

### 추가 모델 다운로드

학습에는 CLIP 이미지 인코더가 필요합니다:

```python
import open_clip
import torch

model, _, _ = open_clip.create_model_and_transforms(
    'xlm-roberta-large-ViT-H-14',
    pretrained='frozen_laion5b_s13b_b90k'
)
visual_state_dict = model.visual.state_dict()
torch.save(
    visual_state_dict,
    '/mnt/models/Wan2.2-S2V-14B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
)
```

```bash
pip install open-clip-torch
```

### 데이터 준비

```
talking_face_data/
├── video_001/
│   ├── sub_clip.mp4          # 원본 비디오 (2-10초)
│   ├── audio.wav             # 16kHz 오디오
│   ├── images/               # 프레임 (25fps)
│   ├── face_masks/           # 얼굴 마스크
│   └── lip_masks/            # 입술 마스크
├── video_002/
└── video_path.txt            # 비디오 경로 목록
```

### 학습 실행

```bash
bash train_lora_14B.sh
```

주요 하이퍼파라미터:

| 파라미터 | 권장값 | 설명 |
|---------|--------|------|
| `learning_rate` | 1e-4 | LoRA 학습률 |
| `num_train_epochs` | 10 | 에폭 수 |
| `gradient_accumulation_steps` | 8 | Gradient 누적 |
| `lora_rank` | 128 | LoRA rank |
| `video_sample_n_frames` | 81 | 프레임 수 (~2초@25fps) |

## Changelog

### 2026-02-09: 듀얼 GPU 추론 및 속도 최적화

**핵심 변경사항:**

- **듀얼 GPU 추론**: T5 인코더를 GPU 1, DiT 트랜스포머를 GPU 0으로 분리하여 메모리 효율 극대화
  - `speech2video.py`: `t5_device_id` 파라미터 추가, GPU 간 텐서 자동 전송
  - 모델 로딩: CPU → state_dict 로드 → GPU 이동 (OOM 방지)
- **속도 최적화**: inference_steps 40→15, infer_frames 40→80 (UniPC 솔버)
- **TeaCache 통합** (실험적): `model_s2v.py`에 TeaCache 지원 추가, 캐시 비교 신호 최적화 진행 중
- **비디오 저장 수정**: `save_videos_grid` + `ffmpeg` subprocess로 오디오 병합
- **서버 설정 업데이트**: 기본값 최적화 (steps=15, infer_frames=80, offload=false)

**환경 설정:**

- flash-attn 2.7.4.post1 설치 (2.8.3은 PyTorch 2.6.0 ABI 호환 불가)
- ffmpeg 설치 (비디오+오디오 병합)

### 2026-02-09: Wan2.1 정리 및 환경 구성

**삭제된 파일 (Wan2.1/1B 전용):**

- Shell scripts: `inference.sh`, `multiple_gpu_inference.sh`, `train_lora.sh`, `train_14B.sh`, `train_14B_lora.sh`, `train_1B_*.sh`
- Python scripts: `inference.py`, `train_1B_rec_vec.py`, `train_1B_rec_vec_lora.py`, `train_1B_square.py`
- Model/pipeline: `wan_fantasy_transformer3d_1B.py`, `vocal_projector_fantasy_1B.py`, `wan_inference_long_pipeline.py`, `wan_inference_pipeline_fantasy.py`, `pipeline_wan_fun_inpaint.py`
- Configs: `wan_t2v_14B.py`, `wan_i2v_14B.py`, `wan_t2v_1_3B.py`, `wan2.1/wan_civitai.yaml`

**경로 업데이트:**

- `app.py`, `server.py`: `CHECKPOINT_DIR` → `/mnt/models/Wan2.2-S2V-14B`
- `start.sh`: Working directory → `/home/ubuntu/WanAvatar`
- `train_lora_14B.sh`: 모든 경로 업데이트

## Credits

- **Wan2.2-S2V-14B**: Alibaba's Speech-to-Video model
- **TeaCache**: Timestep Embedding Aware Cache ([ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache))
- **Flash Attention 2**: Tri Dao's flash-attn
- **StableAvatar**: Frontend inspiration ([StableAvatar](https://github.com/Francis-Rings/StableAvatar))

## License

This project is for research purposes. Please refer to the original model licenses for usage terms.
