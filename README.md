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

### Web Interface

```bash
python app.py
```

The application will start at `http://0.0.0.0:7891`

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
├── app.py                    # Gradio web application
├── generate.py               # CLI generation script
├── inference.py              # Inference utilities
├── wan/
│   ├── speech2video.py       # Main S2V pipeline with custom safetensors loading
│   ├── modules/
│   │   ├── s2v/              # Speech-to-Video model components
│   │   │   ├── model_s2v.py  # WanModel_S2V transformer
│   │   │   ├── audio_encoder.py
│   │   │   └── ...
│   │   └── vae2_1.py         # VAE encoder/decoder
│   ├── configs/              # Model configurations
│   ├── utils/                # Utilities (solvers, optimizations)
│   └── distributed/          # Distributed training utilities
├── examples/                 # Example images for testing
├── requirements.txt          # Base dependencies
├── requirements_wan22.txt    # Wan2.2 specific dependencies
└── requirements_s2v.txt      # S2V specific dependencies
```

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

## Credits

- **Wan2.2-S2V-14B**: Alibaba's Speech-to-Video model
- **StableAvatar**: Original frontend inspiration from [StableAvatar](https://github.com/Francis-Rings/StableAvatar)
- **Gradio**: Web interface framework

## License

This project is for research purposes. Please refer to the original model licenses for usage terms.
