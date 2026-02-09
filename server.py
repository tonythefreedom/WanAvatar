#!/usr/bin/env python3
"""
WanAvatar - FastAPI Backend Server
Based on Wan2.2-S2V-14B model
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
from pathlib import Path
from typing import Optional

import torch
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

# Device setup
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

# Global pipeline
pipeline = None
generation_status = {}


# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(
    title="WanAvatar API",
    description="Speech-to-Video Generation API powered by Wan2.2-S2V-14B",
    version="1.0.0",
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
# Models
# ============================================================================
class GenerateRequest(BaseModel):
    image_path: str
    audio_path: str
    prompt: str = "A person speaking naturally with subtle expressions"
    negative_prompt: str = "blurry, low quality, distorted face"
    resolution: str = "720*1280"
    num_clips: int = 0
    inference_steps: int = 15
    guidance_scale: float = 4.5
    infer_frames: int = 80
    seed: int = -1
    offload_model: bool = False
    use_teacache: bool = True
    teacache_thresh: float = 0.1


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
    global pipeline
    if pipeline is not None:
        return pipeline

    logging.info("Loading Wan2.2-S2V-14B model...")
    cfg = WAN_CONFIGS['s2v-14B']

    num_gpus = torch.cuda.device_count()
    pipeline = wan.WanS2V(
        config=cfg,
        checkpoint_dir=CHECKPOINT_DIR,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        t5_device_id=1 if num_gpus > 1 else 0,
        init_on_cpu=False,
    )

    logging.info("Model loaded successfully!")
    return pipeline


# ============================================================================
# Background Tasks
# ============================================================================
def generate_video_task(task_id: str, params: dict):
    """Background task for video generation (synchronous)."""
    global pipeline, generation_status

    try:
        logging.info(f"Starting generation task: {task_id}")
        generation_status[task_id] = {
            "status": "processing",
            "progress": 0.1,
            "message": "Loading model...",
            "output_path": None,
        }

        logging.info("Loading pipeline...")
        pipeline = load_pipeline()

        logging.info("Pipeline loaded successfully")
        generation_status[task_id]["progress"] = 0.2
        generation_status[task_id]["message"] = "Generating video..."

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
        logging.info(f"Starting generation with resolution {params['resolution']}, max_area={max_area}")

        # Generate
        video = pipeline.generate(
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
            use_teacache=params.get("use_teacache", True),
            teacache_thresh=params.get("teacache_thresh", 0.3),
        )

        generation_status[task_id]["progress"] = 0.8
        generation_status[task_id]["message"] = "Saving video..."

        # Save video
        from wan.utils.utils import save_videos_grid

        video_path = str(OUTPUT_DIR / f"{timestamp}.mp4")
        # video shape: [C, T, H, W] -> add batch dim [1, C, T, H, W]
        save_videos_grid(video[None], video_path, rescale=True, fps=16)

        # Merge audio with ffmpeg
        output_path = str(OUTPUT_DIR / f"{timestamp}_with_audio.mp4")
        import subprocess
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", params["audio_path"],
            "-c:v", "copy", "-c:a", "aac",
            "-shortest", output_path
        ], capture_output=True)

        # Cleanup
        del video
        gc.collect()
        torch.cuda.empty_cache()

        generation_status[task_id] = {
            "status": "completed",
            "progress": 1.0,
            "message": "Generation completed!",
            "output_path": f"/outputs/{timestamp}_with_audio.mp4",
            "seed": seed,
        }

    except Exception as e:
        logging.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        generation_status[task_id] = {
            "status": "failed",
            "progress": 0,
            "message": str(e),
            "output_path": None,
        }


# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/")
async def root():
    """Serve React frontend or API info."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "WanAvatar API", "version": "1.0.0", "docs": "/docs"}


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "device": DEVICE,
        "dtype": str(DTYPE),
        "model_loaded": pipeline is not None,
        "has_moviepy": HAS_MOVIEPY,
        "has_audio_separator": HAS_AUDIO_SEPARATOR,
    }


@app.get("/api/config")
async def get_config():
    return {
        "resolutions": list(SIZE_CONFIGS.keys()),
        "default_resolution": "720*1280",
        "default_steps": 15,
        "default_guidance": 4.5,
        "default_frames": 80,
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
async def generate_video(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Start video generation task."""
    task_id = str(uuid.uuid4())

    generation_status[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Task queued",
        "output_path": None,
    }

    background_tasks.add_task(
        generate_video_task,
        task_id,
        request.dict()
    )

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


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    logging.info("Starting WanAvatar API Server...")
    logging.info(f"Checkpoint directory: {CHECKPOINT_DIR}")
    logging.info(f"Device: {DEVICE}, Dtype: {DTYPE}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
