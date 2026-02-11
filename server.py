#!/usr/bin/env python3
"""
WanAvatar - FastAPI Backend Server
S2V (Speech-to-Video) + I2V (Image-to-Video / SVI 2.0 Pro)
"""
import os
import sys
import gc
import uuid
import asyncio
import logging
import datetime
import random
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import pickle
import torch
import torch.distributed as dist
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Wan2.2 imports
import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS

# Optional imports
try:
    from moviepy.editor import VideoFileClip
    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False

try:
    from audio_separator.separator import Separator
    HAS_AUDIO_SEPARATOR = True
except ImportError:
    HAS_AUDIO_SEPARATOR = False


# ============================================================================
# Configuration
# ============================================================================
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

CHECKPOINT_DIR = "/mnt/models/Wan2.2-S2V-14B"
I2V_CHECKPOINT_DIR = "/mnt/models/Wan2.2-I2V-14B-A"
LORA_CHECKPOINT_DIR = "/home/ubuntu/WanAvatar/output_lora_14B/checkpoint-50"
LORA_STRENGTH = 0.0  # LoRA multiplier (0.0=off, 1.0=full)

# Device setup
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

# Global pipelines
pipeline = None       # S2V pipeline
i2v_pipeline = None   # I2V pipeline
active_model = None   # 's2v' or 'i2v' — tracks which model is on GPU
generation_status = {}
gpu_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)

# Distributed / Sequence Parallel setup
USE_SP = False
LOCAL_RANK = 0
WORLD_SIZE = 1


def setup_distributed():
    """Initialize distributed process group for Sequence Parallel."""
    global USE_SP, LOCAL_RANK, WORLD_SIZE
    if os.environ.get('LOCAL_RANK') is not None:
        LOCAL_RANK = int(os.environ['LOCAL_RANK'])
        WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
        torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(days=7),
        )
        if WORLD_SIZE > 1:
            USE_SP = True
        logging.info(f"Distributed mode: rank={LOCAL_RANK}/{WORLD_SIZE}, SP={USE_SP}")
    else:
        logging.info("Single-process mode (no SP)")


# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(
    title="WanAvatar API",
    description="S2V + I2V Video Generation API",
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Serve React frontend (built static files)
FRONTEND_DIR = Path(__file__).parent / "frontend" / "dist"
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")


# ============================================================================
# Request Models
# ============================================================================
class GenerateRequest(BaseModel):
    image_path: str
    audio_path: str
    prompt: str = "A person speaking naturally with subtle expressions"
    negative_prompt: str = "ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, three legs, extra limbs"
    resolution: str = "720*1280"
    num_clips: int = 0
    inference_steps: int = 25
    guidance_scale: float = 5.5
    infer_frames: int = 80
    seed: int = -1
    offload_model: bool = False
    use_teacache: bool = False
    teacache_thresh: float = 0.15


class I2VGenerateRequest(BaseModel):
    image_path: str
    prompt: str = "A cinematic video with natural motion, high quality"
    negative_prompt: str = "ugly, blurry, low quality, distorted, deformed"
    resolution: str = "720*1280"
    frame_num: int = 81
    inference_steps: int = 40
    guidance_scale: float = 5.0
    shift: float = 5.0
    seed: int = -1
    offload_model: bool = False


class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending, processing, completed, failed
    progress: float
    message: str
    output_path: Optional[str] = None


# ============================================================================
# Model Loading
# ============================================================================
def load_pipeline():
    """Load Wan2.2 S2V pipeline."""
    global pipeline, active_model
    if pipeline is not None:
        if active_model != 's2v':
            pipeline.noise_model.to(f'cuda:{LOCAL_RANK}')
            pipeline.text_encoder.model.to(f'cuda:{LOCAL_RANK}')
            active_model = 's2v'
        return pipeline

    logging.info(f"Loading Wan2.2-S2V-14B model (rank={LOCAL_RANK}, SP={USE_SP})...")
    cfg = WAN_CONFIGS['s2v-14B']

    lora_dir = LORA_CHECKPOINT_DIR if (LORA_STRENGTH > 0 and os.path.exists(LORA_CHECKPOINT_DIR)) else None

    if USE_SP:
        pipeline = wan.WanS2V(
            config=cfg,
            checkpoint_dir=CHECKPOINT_DIR,
            lora_checkpoint_dir=lora_dir,
            lora_strength=LORA_STRENGTH,
            device_id=LOCAL_RANK,
            rank=LOCAL_RANK,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=True,
            t5_cpu=False,
            t5_device_id=LOCAL_RANK,
            init_on_cpu=False,
        )
        dist.barrier()
    else:
        num_gpus = torch.cuda.device_count()
        pipeline = wan.WanS2V(
            config=cfg,
            checkpoint_dir=CHECKPOINT_DIR,
            lora_checkpoint_dir=lora_dir,
            lora_strength=LORA_STRENGTH,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            t5_device_id=1 if num_gpus > 1 else 0,
            init_on_cpu=False,
        )

    active_model = 's2v'
    logging.info(f"S2V model loaded successfully on rank {LOCAL_RANK}!")
    return pipeline


def load_i2v_pipeline():
    """Load Wan2.2 I2V pipeline."""
    global i2v_pipeline, active_model
    if i2v_pipeline is not None:
        if active_model != 'i2v':
            i2v_pipeline.model.to(f'cuda:{LOCAL_RANK}')
            i2v_pipeline.text_encoder.model.to(f'cuda:{LOCAL_RANK}')
            i2v_pipeline.clip.model.to(f'cuda:{LOCAL_RANK}')
            active_model = 'i2v'
        return i2v_pipeline

    logging.info(f"Loading Wan2.2-I2V-14B-A model (rank={LOCAL_RANK}, SP={USE_SP})...")
    cfg = WAN_CONFIGS['i2v-A14B']

    if USE_SP:
        i2v_pipeline = wan.WanI2V(
            config=cfg,
            checkpoint_dir=I2V_CHECKPOINT_DIR,
            device_id=LOCAL_RANK,
            rank=LOCAL_RANK,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=True,
            t5_cpu=False,
            t5_device_id=LOCAL_RANK,
            init_on_cpu=False,
        )
        dist.barrier()
    else:
        num_gpus = torch.cuda.device_count()
        i2v_pipeline = wan.WanI2V(
            config=cfg,
            checkpoint_dir=I2V_CHECKPOINT_DIR,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            t5_device_id=1 if num_gpus > 1 else 0,
            init_on_cpu=False,
        )

    active_model = 'i2v'
    logging.info(f"I2V model loaded successfully on rank {LOCAL_RANK}!")
    return i2v_pipeline


def ensure_model_loaded(model_type: str):
    """Ensure the requested model is on GPU. Swap if necessary."""
    global active_model
    if active_model == model_type:
        return

    logging.info(f"Swapping model: {active_model} -> {model_type}")

    # Offload current model to CPU
    if active_model == 's2v' and pipeline is not None:
        logging.info("Moving S2V model to CPU...")
        pipeline.noise_model.cpu()
        pipeline.text_encoder.model.cpu()
        torch.cuda.empty_cache()
        gc.collect()
    elif active_model == 'i2v' and i2v_pipeline is not None:
        logging.info("Moving I2V model to CPU...")
        i2v_pipeline.model.cpu()
        i2v_pipeline.text_encoder.model.cpu()
        i2v_pipeline.clip.model.cpu()
        torch.cuda.empty_cache()
        gc.collect()

    # Load requested model
    if model_type == 's2v':
        load_pipeline()
    elif model_type == 'i2v':
        load_i2v_pipeline()

    if dist.is_initialized():
        dist.barrier()
    logging.info(f"Model swap complete. Active: {active_model}")


def broadcast_generate_params(params):
    """Broadcast generation params from rank 0 to all ranks."""
    device = f'cuda:{LOCAL_RANK}'
    if LOCAL_RANK == 0:
        data = pickle.dumps(params)
        size = torch.tensor([len(data)], dtype=torch.long, device=device)
        dist.broadcast(size, src=0)
        data_tensor = torch.tensor(list(data), dtype=torch.uint8, device=device)
        dist.broadcast(data_tensor, src=0)
        return params
    else:
        size = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(size, src=0)
        data_tensor = torch.empty(size.item(), dtype=torch.uint8, device=device)
        dist.broadcast(data_tensor, src=0)
        return pickle.loads(data_tensor.cpu().numpy().tobytes())


def worker_loop():
    """Worker loop for non-master ranks in SP mode."""
    global pipeline, i2v_pipeline
    logging.info(f"Rank {LOCAL_RANK}: Entering worker loop...")

    while True:
        # Wait for command from rank 0
        cmd = torch.tensor([0], dtype=torch.long, device=f'cuda:{LOCAL_RANK}')
        dist.broadcast(cmd, src=0)

        if cmd.item() == 1:  # S2V Generate
            params = broadcast_generate_params(None)
            logging.info(f"Rank {LOCAL_RANK}: Starting S2V generation...")
            try:
                pipeline.generate(**params)
                logging.info(f"Rank {LOCAL_RANK}: S2V generation done.")
            except Exception as e:
                logging.error(f"Rank {LOCAL_RANK}: S2V generation error: {e}")
                import traceback
                traceback.print_exc()

        elif cmd.item() == 2:  # I2V Generate
            params = broadcast_generate_params(None)
            logging.info(f"Rank {LOCAL_RANK}: Swapping to I2V model...")
            try:
                ensure_model_loaded('i2v')
                # Load image from path on each rank
                image_path = params.pop('_image_path')
                params['img'] = Image.open(image_path).convert('RGB')
                logging.info(f"Rank {LOCAL_RANK}: Starting I2V generation...")
                i2v_pipeline.generate(**params)
                logging.info(f"Rank {LOCAL_RANK}: I2V generation done.")
            except Exception as e:
                logging.error(f"Rank {LOCAL_RANK}: I2V generation error: {e}")
                import traceback
                traceback.print_exc()

        elif cmd.item() == -1:  # Shutdown
            logging.info(f"Rank {LOCAL_RANK}: Shutting down.")
            break


# ============================================================================
# Background Tasks
# ============================================================================
def generate_video_task(task_id: str, params: dict):
    """Background task for S2V video generation (runs in thread pool)."""
    global pipeline, generation_status

    if not gpu_lock.acquire(blocking=False):
        generation_status[task_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Waiting for GPU (another generation in progress)...",
            "output_path": None,
        }
        gpu_lock.acquire()  # block until GPU is free

    try:
        logging.info(f"Starting S2V generation task: {task_id}")
        generation_status[task_id] = {
            "status": "processing",
            "progress": 0.02,
            "message": "Switching to S2V model...",
            "output_path": None,
        }

        ensure_model_loaded('s2v')

        generation_status[task_id]["progress"] = 0.1
        generation_status[task_id]["message"] = "Preparing generation..."

        # Parse parameters
        seed = params["seed"]
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        # Calculate max_area from resolution string (height*width)
        try:
            res_parts = params["resolution"].split("*")
            res_height, res_width = int(res_parts[0]), int(res_parts[1])
            max_area = res_height * res_width
        except:
            max_area = MAX_AREA_CONFIGS.get(params["resolution"], 720 * 1280)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.info(f"Starting S2V generation with resolution {params['resolution']}, max_area={max_area}")

        # Progress callback for real-time updates
        def progress_callback(progress, message):
            generation_status[task_id]["progress"] = round(progress, 3)
            generation_status[task_id]["message"] = message

        # Build generation kwargs
        gen_kwargs = dict(
            input_prompt=params["prompt"],
            ref_image_path=params["image_path"],
            audio_path=params["audio_path"],
            enable_tts=False,
            tts_prompt_audio=None,
            tts_prompt_text=None,
            tts_text=None,
            num_repeat=params["num_clips"] if params["num_clips"] > 0 else None,
            pose_video=None,
            max_area=max_area,
            infer_frames=params["infer_frames"],
            shift=3.0,
            sample_solver='unipc',
            sampling_steps=params["inference_steps"],
            guide_scale=params["guidance_scale"],
            n_prompt=params["negative_prompt"],
            seed=seed,
            offload_model=params["offload_model"],
            init_first_frame=False,
            use_teacache=params.get("use_teacache", False),
            teacache_thresh=params.get("teacache_thresh", 0.15),
        )

        # Signal SP workers to start S2V generation
        if USE_SP:
            cmd = torch.tensor([1], dtype=torch.long, device=f'cuda:{LOCAL_RANK}')
            dist.broadcast(cmd, src=0)
            broadcast_generate_params(gen_kwargs)

        # Generate (rank 0 gets progress callback, workers don't)
        video = pipeline.generate(**gen_kwargs, progress_callback=progress_callback)

        generation_status[task_id]["progress"] = 0.85
        generation_status[task_id]["message"] = "Saving video..."

        # Save video
        from wan.utils.utils import save_videos_grid

        video_path = str(OUTPUT_DIR / f"{timestamp}.mp4")
        save_videos_grid(video[None], video_path, rescale=True, fps=16)

        generation_status[task_id]["progress"] = 0.9
        generation_status[task_id]["message"] = "Merging audio..."

        # Merge audio with ffmpeg
        output_with_audio = str(OUTPUT_DIR / f"{timestamp}_with_audio.mp4")
        import subprocess
        try:
            result = subprocess.run([
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", params["audio_path"],
                "-c:v", "copy", "-c:a", "aac",
                "-shortest", output_with_audio
            ], capture_output=True, timeout=120)

            if result.returncode == 0 and os.path.exists(output_with_audio) and os.path.getsize(output_with_audio) > 0:
                final_path = f"/outputs/{timestamp}_with_audio.mp4"
                logging.info(f"Audio merged successfully: {output_with_audio}")
            else:
                logging.warning(f"ffmpeg audio merge failed (returncode={result.returncode}): {result.stderr.decode()}")
                final_path = f"/outputs/{timestamp}.mp4"
        except Exception as merge_err:
            logging.warning(f"ffmpeg audio merge error: {merge_err}")
            final_path = f"/outputs/{timestamp}.mp4"

        # Cleanup
        del video
        gc.collect()
        torch.cuda.empty_cache()

        generation_status[task_id] = {
            "status": "completed",
            "progress": 1.0,
            "message": "Generation completed!",
            "output_path": final_path,
            "seed": seed,
        }

    except Exception as e:
        logging.error(f"S2V generation failed: {e}")
        import traceback
        traceback.print_exc()
        generation_status[task_id] = {
            "status": "failed",
            "progress": 0,
            "message": str(e),
            "output_path": None,
        }
    finally:
        gpu_lock.release()


def generate_i2v_task(task_id: str, params: dict):
    """Background task for I2V video generation."""
    global i2v_pipeline, generation_status

    if not gpu_lock.acquire(blocking=False):
        generation_status[task_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Waiting for GPU (another generation in progress)...",
            "output_path": None,
        }
        gpu_lock.acquire()

    try:
        logging.info(f"Starting I2V generation task: {task_id}")
        generation_status[task_id] = {
            "status": "processing",
            "progress": 0.02,
            "message": "Switching to I2V model...",
            "output_path": None,
        }

        ensure_model_loaded('i2v')

        generation_status[task_id]["progress"] = 0.1
        generation_status[task_id]["message"] = "Preparing I2V generation..."

        seed = params["seed"]
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        # Parse resolution
        try:
            res_parts = params["resolution"].split("*")
            res_height, res_width = int(res_parts[0]), int(res_parts[1])
            max_area = res_height * res_width
        except:
            max_area = 720 * 1280

        # Load input image
        img = Image.open(params["image_path"]).convert("RGB")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.info(f"Starting I2V generation with resolution {params['resolution']}, max_area={max_area}")

        def progress_callback(progress, message):
            generation_status[task_id]["progress"] = round(progress, 3)
            generation_status[task_id]["message"] = message

        gen_kwargs = dict(
            input_prompt=params["prompt"],
            img=img,
            max_area=max_area,
            frame_num=params["frame_num"],
            shift=params["shift"],
            sample_solver='unipc',
            sampling_steps=params["inference_steps"],
            guide_scale=params["guidance_scale"],
            n_prompt=params["negative_prompt"],
            seed=seed,
            offload_model=params["offload_model"],
        )

        # Signal SP workers to start I2V generation
        if USE_SP:
            cmd = torch.tensor([2], dtype=torch.long, device=f'cuda:{LOCAL_RANK}')
            dist.broadcast(cmd, src=0)
            # Send kwargs without PIL Image — workers load from path
            sp_kwargs = {k: v for k, v in gen_kwargs.items() if k != 'img'}
            sp_kwargs['_image_path'] = params["image_path"]
            broadcast_generate_params(sp_kwargs)

        video = i2v_pipeline.generate(**gen_kwargs, progress_callback=progress_callback)

        generation_status[task_id]["progress"] = 0.9
        generation_status[task_id]["message"] = "Saving video..."

        from wan.utils.utils import save_videos_grid
        video_path = str(OUTPUT_DIR / f"i2v_{timestamp}.mp4")
        save_videos_grid(video[None], video_path, rescale=True, fps=16)

        final_path = f"/outputs/i2v_{timestamp}.mp4"

        del video
        gc.collect()
        torch.cuda.empty_cache()

        generation_status[task_id] = {
            "status": "completed",
            "progress": 1.0,
            "message": "I2V generation completed!",
            "output_path": final_path,
            "seed": seed,
        }

    except Exception as e:
        logging.error(f"I2V generation failed: {e}")
        import traceback
        traceback.print_exc()
        generation_status[task_id] = {
            "status": "failed",
            "progress": 0,
            "message": str(e),
            "output_path": None,
        }
    finally:
        gpu_lock.release()


# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/")
async def root():
    """Serve React frontend or API info."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "WanAvatar API", "version": "2.0.0", "docs": "/docs"}


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "device": DEVICE,
        "dtype": str(DTYPE),
        "model_loaded": pipeline is not None,
        "i2v_model_loaded": i2v_pipeline is not None,
        "active_model": active_model,
        "lora_enabled": LORA_STRENGTH > 0 and os.path.exists(LORA_CHECKPOINT_DIR),
        "lora_strength": LORA_STRENGTH,
        "gpu_busy": gpu_lock.locked(),
        "sequence_parallel": USE_SP,
        "world_size": WORLD_SIZE,
        "i2v_available": os.path.exists(os.path.join(I2V_CHECKPOINT_DIR, "low_noise_model", "config.json")),
        "has_moviepy": HAS_MOVIEPY,
        "has_audio_separator": HAS_AUDIO_SEPARATOR,
    }


@app.get("/api/config")
async def get_config():
    return {
        "resolutions": list(SIZE_CONFIGS.keys()),
        "default_resolution": "720*1280",
        "default_steps": 25,
        "default_guidance": 5.5,
        "default_frames": 80,
        "default_use_teacache": False,
        "default_teacache_thresh": 0.15,
        # I2V defaults
        "i2v_available": os.path.exists(os.path.join(I2V_CHECKPOINT_DIR, "low_noise_model", "config.json")),
        "i2v_default_steps": 40,
        "i2v_default_guidance": 5.0,
        "i2v_default_frame_num": 81,
        "i2v_default_shift": 5.0,
    }


@app.post("/api/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload reference image."""
    try:
        ext = Path(file.filename).suffix.lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            raise HTTPException(400, "Invalid image format")

        filename = f"{uuid.uuid4()}{ext}"
        filepath = UPLOAD_DIR / filename

        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)

        # Get image dimensions
        with Image.open(filepath) as img:
            width, height = img.size

        return {
            "path": str(filepath),
            "url": f"/uploads/{filename}",
            "width": width,
            "height": height,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/upload/audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload driving audio."""
    try:
        ext = Path(file.filename).suffix.lower()
        if ext not in [".wav", ".mp3", ".ogg", ".flac"]:
            raise HTTPException(400, "Invalid audio format")

        filename = f"{uuid.uuid4()}{ext}"
        filepath = UPLOAD_DIR / filename

        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)

        return {"path": str(filepath), "url": f"/uploads/{filename}"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/upload/video")
async def upload_video(file: UploadFile = File(...)):
    """Upload video for audio extraction."""
    try:
        ext = Path(file.filename).suffix.lower()
        if ext not in [".mp4", ".avi", ".mov", ".webm"]:
            raise HTTPException(400, "Invalid video format")

        filename = f"{uuid.uuid4()}{ext}"
        filepath = UPLOAD_DIR / filename

        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)

        return {"path": str(filepath), "url": f"/uploads/{filename}"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/generate")
async def generate_video(request: GenerateRequest):
    """Start S2V video generation task."""
    task_id = str(uuid.uuid4())

    generation_status[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Task queued",
        "output_path": None,
    }

    executor.submit(generate_video_task, task_id, request.dict())

    return {"task_id": task_id}


@app.post("/api/generate-i2v")
async def generate_i2v(request: I2VGenerateRequest):
    """Start I2V video generation task."""
    i2v_model_path = os.path.join(I2V_CHECKPOINT_DIR, "low_noise_model", "config.json")
    if not os.path.exists(i2v_model_path):
        raise HTTPException(503, "I2V model not available. Download Wan2.2-I2V-A14B first.")

    task_id = str(uuid.uuid4())

    generation_status[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "I2V task queued",
        "output_path": None,
    }

    executor.submit(generate_i2v_task, task_id, request.dict())

    return {"task_id": task_id}


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """Get generation task status."""
    if task_id not in generation_status:
        raise HTTPException(404, "Task not found")

    return generation_status[task_id]


@app.post("/api/extract-audio")
async def extract_audio(video_path: str = Form(...)):
    """Extract audio from video."""
    if not HAS_MOVIEPY:
        raise HTTPException(500, "moviepy not installed")

    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(OUTPUT_DIR / f"{timestamp}_extracted.wav")

        video = VideoFileClip(video_path)
        video.audio.write_audiofile(output_path, codec='pcm_s16le')
        video.close()

        return {
            "path": output_path,
            "url": f"/outputs/{timestamp}_extracted.wav",
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/separate-vocals")
async def separate_vocals(audio_path: str = Form(...)):
    """Separate vocals from audio."""
    if not HAS_AUDIO_SEPARATOR:
        raise HTTPException(500, "audio-separator not installed")

    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = OUTPUT_DIR / f"temp_{timestamp}"
        temp_dir.mkdir(exist_ok=True)

        separator = Separator(
            output_dir=str(temp_dir),
            output_single_stem="vocals",
        )
        separator.load_model("Kim_Vocal_2.onnx")
        outputs = separator.separate(audio_path)

        vocal_file = temp_dir / outputs[0]
        output_path = OUTPUT_DIR / f"{timestamp}_vocals.wav"

        shutil.move(str(vocal_file), str(output_path))
        shutil.rmtree(temp_dir, ignore_errors=True)

        return {
            "path": str(output_path),
            "url": f"/outputs/{timestamp}_vocals.wav",
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/uploads/images")
async def list_uploaded_images():
    """List all uploaded images."""
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
    images = []
    for f in sorted(UPLOAD_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if f.suffix.lower() in IMAGE_EXTS:
            stat = f.stat()
            try:
                with Image.open(f) as img:
                    width, height = img.size
            except Exception:
                width, height = 0, 0
            images.append({
                "filename": f.name,
                "url": f"/uploads/{f.name}",
                "path": str(f),
                "width": width,
                "height": height,
                "size": stat.st_size,
                "created_at": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
    return {"images": images, "total": len(images)}


@app.get("/api/uploads/audio")
async def list_uploaded_audio():
    """List all uploaded audio files."""
    AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac"}
    audio_files = []
    for f in sorted(UPLOAD_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if f.suffix.lower() in AUDIO_EXTS:
            stat = f.stat()
            audio_files.append({
                "filename": f.name,
                "url": f"/uploads/{f.name}",
                "path": str(f),
                "size": stat.st_size,
                "created_at": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
    return {"audio": audio_files, "total": len(audio_files)}


@app.get("/api/videos")
async def list_videos():
    """List all generated videos with metadata."""
    videos = []
    for f in sorted(OUTPUT_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if f.suffix == ".mp4":
            stat = f.stat()
            videos.append({
                "filename": f.name,
                "url": f"/outputs/{f.name}",
                "size": stat.st_size,
                "created_at": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
    return {"videos": videos, "total": len(videos)}


@app.delete("/api/videos/{filename}")
async def delete_video(filename: str):
    """Delete a generated video."""
    filepath = OUTPUT_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, "Video not found")
    if not filepath.suffix == ".mp4":
        raise HTTPException(400, "Invalid file type")
    # Security: ensure path is within OUTPUT_DIR
    if not filepath.resolve().parent == OUTPUT_DIR.resolve():
        raise HTTPException(403, "Access denied")
    filepath.unlink()
    return {"message": f"Deleted {filename}"}


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    # Initialize distributed if launched with torchrun
    setup_distributed()

    logging.info("Starting WanAvatar API Server...")
    logging.info(f"S2V checkpoint: {CHECKPOINT_DIR}")
    logging.info(f"I2V checkpoint: {I2V_CHECKPOINT_DIR} (available={os.path.exists(I2V_CHECKPOINT_DIR)})")
    logging.info(f"Device: {DEVICE}, Dtype: {DTYPE}")
    logging.info(f"Sequence Parallel: {USE_SP}, World Size: {WORLD_SIZE}")

    # Eager-load S2V pipeline so all ranks load together
    logging.info("Loading S2V pipeline at startup...")
    load_pipeline()
    logging.info("S2V pipeline ready!")

    if USE_SP and LOCAL_RANK != 0:
        # Non-master ranks enter worker loop
        worker_loop()
    else:
        # Master rank (or single-process) serves the API
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
        )
