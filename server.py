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
import subprocess
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import json as _json
import requests as http_requests
import websocket as ws_client

import jwt as pyjwt
import aiosqlite
import bcrypt as _bcrypt

import pickle
import torch
import torch.distributed as dist
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Depends, Request
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
BACKGROUNDS_DIR = Path("background/stages")
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
BACKGROUNDS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = "/mnt/models/Wan2.2-S2V-14B"
I2V_CHECKPOINT_DIR = "/mnt/models/Wan2.2-I2V-14B-A"
LORA_CHECKPOINT_DIR = "/home/ubuntu/WanAvatar/output_lora_14B/checkpoint-50"
LORA_STRENGTH = 0.0  # LoRA multiplier (0.0=off, 1.0=full)

# I2V GGUF configuration (diffusers pipeline with Q4_K_M quantization)
I2V_DIFFUSERS_DIR = "/mnt/models/Wan2.2-I2V-A14B-Diffusers"
I2V_GGUF_DIR = "/mnt/models/Wan2.2-I2V-A14B-GGUF"
I2V_GGUF_HIGH = os.path.join(I2V_GGUF_DIR, "HighNoise", "Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf")
I2V_GGUF_LOW = os.path.join(I2V_GGUF_DIR, "LowNoise", "Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf")

# FLUX configuration
FLUX_MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
FLUX_CACHE_DIR = "/mnt/models"
REALESRGAN_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "RealESRGAN_x2plus.pth")

# ComfyUI configuration
COMFYUI_DIR = Path("/home/ubuntu/ComfyUI")
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
COMFYUI_WS_URL = os.environ.get("COMFYUI_WS_URL", "ws://127.0.0.1:8188/ws")
WORKFLOW_DIR = Path(__file__).parent / "workflow"

# YouTube / Slack configuration
YOUTUBE_CLIENT_SECRET = Path("client_secret.json")
YOUTUBE_TOKEN = Path("token.json")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_NOTIFY_EMAIL = os.environ.get("SLACK_NOTIFY_EMAIL")

# Auth / DB configuration
DB_PATH = Path("wanavatardb.sqlite3")
JWT_SECRET = os.environ.get("JWT_SECRET", "fallback-dev-secret")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 168  # 7 days
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
SUPER_ADMIN_EMAIL = os.environ.get("SUPER_ADMIN", "")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASS", "")

# Workflow registry - defines available workflows and their input mappings
WORKFLOW_REGISTRY = {
    "change_character": {
        "id": "change_character",
        "display_name": {"en": "Change Character V1.1", "ko": "캐릭터 변경 V1.1", "zh": "角色替换 V1.1"},
        "description": {
            "en": "Replace a character in a reference video with your character image. Single front-facing photo needed.",
            "ko": "참조 비디오의 캐릭터를 사용자 이미지로 교체합니다. 정면 사진 1장 필요.",
            "zh": "将参考视频中的角色替换为您的角色图片。只需一张正面照片。",
        },
        "api_json": "Change_character_V1.1_api.json",
        "inputs": [
            {"key": "ref_image", "type": "image", "node_id": "91", "field": "inputs.image",
             "upload_to_comfyui": True, "required": True, "avatar_gallery": True,
             "label": {"en": "Reference Image (Character)", "ko": "참조 이미지 (캐릭터)", "zh": "参考图片（角色）"}},
            {"key": "ref_video", "type": "video", "node_id": "114", "field": "inputs.video",
             "upload_to_comfyui": True, "required": True, "allow_youtube": True,
             "label": {"en": "Reference Video (Motion)", "ko": "참조 비디오 (모션)", "zh": "参考视频（动作）"}},
            {"key": "prompt", "type": "text", "node_id": "209", "field": "inputs.positive_prompt",
             "default": "The character is dancing in the room", "rows": 4,
             "label": {"en": "Scene Description", "ko": "장면 설명", "zh": "场景描述"}},
            {"key": "aspect_ratio", "type": "select_buttons",
             "node_ids": {"width": "123", "height": "124"},
             "fields": {"width": "inputs.value", "height": "inputs.value"},
             "default": "portrait",
             "label": {"en": "Aspect Ratio", "ko": "비율", "zh": "比例"},
             "options": [
                 {"value": "portrait", "label": {"en": "Portrait (9:16)", "ko": "세로 (9:16)", "zh": "竖屏 (9:16)"}, "params": {"width": 480, "height": 832}},
                 {"value": "landscape", "label": {"en": "Landscape (16:9)", "ko": "가로 (16:9)", "zh": "横屏 (16:9)"}, "params": {"width": 832, "height": 480}},
                 {"value": "square", "label": {"en": "Square (1:1)", "ko": "정사각 (1:1)", "zh": "正方形 (1:1)"}, "params": {"width": 640, "height": 640}},
             ]},
            {"key": "bg_image", "type": "image", "node_id": "__custom_bg__",
             "field": "inputs.image", "upload_to_comfyui": True, "required": False,
             "background_gallery": True,
             "default_path": str(BACKGROUNDS_DIR / "dance_stage_01.jpg"),
             "default_preview": "/background/stages/dance_stage_01.jpg",
             "label": {"en": "Background Image (Optional)", "ko": "배경 이미지 (선택)", "zh": "背景图片（可选）"}},
            {"key": "bg_prompt", "type": "text", "node_id": "__bg_prompt__",
             "field": "inputs.positive_prompt", "default": "", "rows": 3, "required": False,
             "label": {"en": "Background Description (Optional)", "ko": "배경 설명 (선택)", "zh": "背景描述（可选）"}},
            {"key": "custom_audio", "type": "audio", "node_id": "__custom_audio__",
             "field": "inputs.audio", "required": False,
             "label": {"en": "Replace Audio (Optional)", "ko": "오디오 교체 (선택)", "zh": "替换音频（可选）"}},
        ],
    },
    "fashion_change": {
        "id": "fashion_change",
        "display_name": {"en": "Fashion Change", "ko": "패션 체인지", "zh": "时尚换装"},
        "description": {
            "en": "Change avatar clothing using FLUX Klein inpainting. Auto-masks clothing area and replaces with new fashion style.",
            "ko": "FLUX Klein 인페인팅으로 아바타 의상을 변경합니다. 옷 영역을 자동 마스킹하고 새로운 패션 스타일로 교체합니다.",
            "zh": "使用FLUX Klein修复功能更换角色服装。自动遮罩服装区域并替换为新的时尚风格。",
        },
        "api_json": "flux_klein_fashion_api.json",
        "output_type": "image",
        "inputs": [
            {"key": "avatar_image", "type": "image", "node_id": "76", "field": "inputs.image",
             "upload_to_comfyui": True, "required": True, "avatar_gallery": True,
             "label": {"en": "Avatar Image", "ko": "아바타 이미지", "zh": "角色图片"}},
            {"key": "clothing_ref", "type": "image", "node_id": "105", "field": "inputs.image",
             "upload_to_comfyui": True, "required": False,
             "label": {"en": "Clothing Reference (Optional)", "ko": "의류 참조 이미지 (선택)", "zh": "服装参考图片（可选）"}},
            {"key": "fashion_prompt", "type": "text", "node_id": "106:74", "field": "inputs.text",
             "default": "Oversized graphic t-shirt with biker shorts and chunky sneakers",
             "rows": 3,
             "label": {"en": "Fashion Description", "ko": "패션 설명", "zh": "时尚描述"}},
            {"key": "fashion_style", "type": "fashion_select",
             "csv_path": "settings/fashion_hair/s1.csv",
             "label": {"en": "Fashion Style Presets", "ko": "패션 스타일 프리셋", "zh": "时尚风格预设"}},
            {"key": "seed", "type": "number", "node_id": "113", "field": "inputs.value",
             "default": -1, "min": -1, "max": 2147483647, "step": 1,
             "label": {"en": "Seed (-1 = random)", "ko": "시드 (-1 = 랜덤)", "zh": "种子 (-1 = 随机)"}},
            {"key": "scene_background", "type": "image", "required": False,
             "upload_to_comfyui": True,
             "label": {"en": "Scene Background (Optional)", "ko": "배경 장면 (선택)", "zh": "场景背景（可选）"},
             "description": {"en": "If set, result will be composited into this background with perspective", "ko": "설정 시 결과물이 이 배경에 원근감 있게 합성됩니다", "zh": "设置后，结果将带有透视效果合成到此背景中"}},
            {"key": "scene_prompt", "type": "text", "required": False,
             "default": "A person standing naturally in the scene, photorealistic, natural lighting, correct perspective and depth, 4K",
             "rows": 2,
             "label": {"en": "Scene Description", "ko": "장면 설명", "zh": "场景描述"},
             "description": {"en": "Prompt for scene composite (used when Scene Background is set)", "ko": "배경 합성 프롬프트 (배경 장면 설정 시 사용)", "zh": "场景合成提示（设置场景背景时使用）"}},
        ],
    },
    "face_swap": {
        "id": "face_swap",
        "display_name": {"en": "Face Swap", "ko": "얼굴 교체", "zh": "换脸"},
        "description": {
            "en": "Swap face/head onto a target body image using BFS LoRA. Preserves clothing, lighting, and background from the target.",
            "ko": "BFS LoRA를 사용하여 대상 이미지에 얼굴/머리를 교체합니다. 대상의 의상, 조명, 배경을 보존합니다.",
            "zh": "使用BFS LoRA将面部/头部换到目标图像上。保留目标的服装、灯光和背景。",
        },
        "api_json": "flux_klein_faceswap_api.json",
        "output_type": "image",
        "inputs": [
            {"key": "avatar_face", "type": "image", "node_id": "11", "field": "inputs.image",
             "upload_to_comfyui": True, "required": True, "avatar_gallery": True,
             "label": {"en": "Avatar (Face to Keep)", "ko": "아바타 (유지할 얼굴)", "zh": "头像（保留面部）"},
             "description": {"en": "Avatar whose face/head to preserve", "ko": "얼굴/머리를 유지할 아바타", "zh": "保留面部/头部的头像"}},
            {"key": "style_source", "type": "image", "node_id": "10", "field": "inputs.image",
             "upload_to_comfyui": True, "required": True, "large_viewer": True,
             "label": {"en": "Style Source (Body/Clothing)", "ko": "스타일 소스 (몸/의상)", "zh": "风格来源（身体/服装）"},
             "description": {"en": "Image whose body/clothing/background to use", "ko": "몸/의상/배경을 사용할 이미지", "zh": "使用其身体/服装/背景的图片"}},
            {"key": "ethnicity", "type": "select", "default": "korean",
             "options": [
                 {"value": "korean", "label": {"en": "Korean", "ko": "한국인", "zh": "韩国人"}},
                 {"value": "asian", "label": {"en": "Asian", "ko": "아시아인", "zh": "亚洲人"}},
                 {"value": "western", "label": {"en": "Western", "ko": "서양인", "zh": "西方人"}},
                 {"value": "auto", "label": {"en": "Auto (from prompt)", "ko": "자동 (프롬프트)", "zh": "自动（从提示）"}},
             ],
             "label": {"en": "Ethnicity", "ko": "인종", "zh": "种族"}},
            {"key": "prompt", "type": "text", "node_id": "50", "field": "inputs.text",
             "default": "head_swap: Use image 1 as the base image, preserving its environment, background, camera perspective, framing, exposure, contrast, and lighting. Remove the head from image 1 and seamlessly replace it with the head from image 2. Match the original head size, face-to-body ratio, neck thickness, shoulder alignment, and camera distance so proportions remain natural and unchanged. Adapt the inserted head to the lighting of image 1 by matching light direction, intensity, softness, color temperature, shadows, and highlights, with no independent relighting. Preserve the identity of image 2, including hair texture, eye color, nose structure, facial proportions, and skin details. Match the pose and expression from image 1, including head tilt, rotation, eye direction, gaze, micro-expressions, and lip position. Ensure seamless neck and jaw blending, consistent skin tone, realistic shadow contact, natural skin texture, and uniform sharpness. Photorealistic, high quality, sharp details, 4K.",
             "rows": 4,
             "label": {"en": "Swap Instruction", "ko": "교체 지시", "zh": "换脸指令"}},
            {"key": "lora_strength", "type": "number", "node_id": "21", "field": "inputs.strength_model",
             "default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05,
             "label": {"en": "LoRA Strength", "ko": "LoRA 강도", "zh": "LoRA 强度"}},
            {"key": "cfg", "type": "number", "node_id": "94", "field": "inputs.cfg",
             "default": 3.5, "min": 1.0, "max": 10.0, "step": 0.5,
             "label": {"en": "CFG Scale", "ko": "CFG 스케일", "zh": "CFG 比例"}},
            {"key": "steps", "type": "number", "node_id": "90", "field": "inputs.steps",
             "default": 20, "min": 4, "max": 50, "step": 1,
             "label": {"en": "Steps", "ko": "스텝", "zh": "步数"}},
            {"key": "seed", "type": "number", "node_id": "92", "field": "inputs.value",
             "default": -1, "min": -1, "max": 2147483647, "step": 1,
             "label": {"en": "Seed (-1 = random)", "ko": "시드 (-1 = 랜덤)", "zh": "种子 (-1 = 随机)"}},
        ],
    },
    "scene_composite": {
        "id": "scene_composite",
        "display_name": {"en": "Scene Composite", "ko": "배경 합성", "zh": "场景合成"},
        "description": {
            "en": "Place a character into a background scene with natural perspective and lighting using Z-Image-Turbo + ControlNet.",
            "ko": "Z-Image-Turbo + ControlNet을 사용하여 캐릭터를 배경에 자연스러운 원근감과 조명으로 배치합니다.",
            "zh": "使用Z-Image-Turbo + ControlNet将角色自然地放入背景场景中。",
        },
        "api_json": "z_image_scene_api.json",
        "output_type": "image",
        "inputs": [
            {"key": "character", "type": "image", "node_id": "10", "field": "inputs.image",
             "upload_to_comfyui": True, "required": True, "avatar_gallery": True,
             "label": {"en": "Character Image", "ko": "캐릭터 이미지", "zh": "角色图片"},
             "description": {"en": "Avatar or character to place in the scene", "ko": "배경에 배치할 아바타 또는 캐릭터", "zh": "要放入场景中的角色"}},
            {"key": "background", "type": "image", "node_id": "11", "field": "inputs.image",
             "upload_to_comfyui": True, "required": True,
             "label": {"en": "Background Scene", "ko": "배경 이미지", "zh": "背景场景"},
             "description": {"en": "Background/scene image for perspective and structure reference", "ko": "원근감과 구조 참조용 배경/장면 이미지", "zh": "用于透视和结构参考的背景/场景图片"}},
            {"key": "prompt", "type": "text", "node_id": "50", "field": "inputs.user_prompt",
             "default": "A person standing naturally in the scene, photorealistic, natural lighting, correct perspective and depth, 4K",
             "rows": 3,
             "label": {"en": "Scene Description", "ko": "장면 설명", "zh": "场景描述"}},
            {"key": "negative_prompt", "type": "text", "node_id": "51", "field": "inputs.user_prompt",
             "default": "blurry, low quality, deformed, bad anatomy, watermark, text, extra limbs",
             "rows": 2,
             "label": {"en": "Negative Prompt", "ko": "네거티브 프롬프트", "zh": "负面提示"}},
            {"key": "cn_strength", "type": "number", "node_id": "56", "field": "inputs.strength",
             "default": 0.8, "min": 0.1, "max": 1.5, "step": 0.05,
             "label": {"en": "ControlNet Strength", "ko": "ControlNet 강도", "zh": "ControlNet 强度"}},
            {"key": "steps", "type": "number", "node_id": "90", "field": "inputs.steps",
             "default": 9, "min": 4, "max": 30, "step": 1,
             "label": {"en": "Steps", "ko": "스텝", "zh": "步数"}},
            {"key": "cfg", "type": "number", "node_id": "90", "field": "inputs.cfg",
             "default": 4.0, "min": 1.0, "max": 10.0, "step": 0.5,
             "label": {"en": "CFG Scale", "ko": "CFG 스케일", "zh": "CFG 比例"}},
            {"key": "seed", "type": "number", "node_id": "90", "field": "inputs.seed",
             "default": -1, "min": -1, "max": 2147483647, "step": 1,
             "label": {"en": "Seed (-1 = random)", "ko": "시드 (-1 = 랜덤)", "zh": "种子 (-1 = 随机)"}},
        ],
    },
    "wan_infinitalk": {
        "id": "wan_infinitalk",
        "display_name": {"en": "InfiniTalk (Unlimited Duration)", "ko": "인피니톡 (무제한 길이)", "zh": "InfiniTalk (无限时长)"},
        "description": {
            "en": "Image + Audio → unlimited duration narration/presentation video using Wan2.2 I2V + InfiniTalk patch.",
            "ko": "이미지 + 오디오 → 무제한 길이 나레이션/프리젠테이션 비디오 (Wan2.2 I2V + InfiniTalk 패치).",
            "zh": "图片 + 音频 → 无限时长旁白/演示视频 (Wan2.2 I2V + InfiniTalk 补丁)。",
        },
        "api_json": "wan_infinitalk_api.json",
        "inputs": [
            {"key": "image", "type": "image", "node_id": "97", "field": "inputs.image",
             "upload_to_comfyui": True, "required": True,
             "label": {"en": "Character/Scene Image", "ko": "캐릭터/장면 이미지", "zh": "角色/场景图片"}},
            {"key": "audio", "type": "audio", "node_id": "168", "field": "inputs.audio",
             "upload_to_comfyui": True, "required": True,
             "label": {"en": "Narration Audio (WAV/MP3)", "ko": "나레이션 오디오 (WAV/MP3)", "zh": "旁白音频 (WAV/MP3)"}},
            {"key": "prompt", "type": "text", "node_id": "215", "field": "inputs.prompt",
             "default": "A person speaking naturally, clear expression, professional lighting", "rows": 4,
             "label": {"en": "Scene Description", "ko": "장면 설명", "zh": "场景描述"}},
            {"key": "length", "type": "number", "node_id": "206", "field": "inputs.value",
             "default": 160, "min": 40, "max": 2000, "step": 1,
             "label": {"en": "Frame Count (20fps)", "ko": "프레임 수 (20fps)", "zh": "帧数 (20fps)"},
             "description": {"en": "Auto-calculated from audio: seconds × 20", "ko": "오디오에서 자동 계산: 초 × 20", "zh": "从音频自动计算: 秒 × 20"}},
            {"key": "resolution", "type": "number", "node_id": "216", "field": "inputs.value",
             "default": 1024, "min": 512, "max": 2048, "step": 64,
             "label": {"en": "Long-edge Resolution", "ko": "장변 해상도", "zh": "长边分辨率"}},
        ],
    },
    "fflf_auto_v2": {
        "id": "fflf_auto_v2",
        "display_name": {"en": "FFLF Auto Loop V2", "ko": "FFLF 자동 루프 V2", "zh": "FFLF 自动循环 V2"},
        "description": {
            "en": "Generate a looping transition video from an image sequence using dual-pass sampling.",
            "ko": "이미지 시퀀스에서 듀얼 패스 샘플링으로 루핑 전환 비디오를 생성합니다.",
            "zh": "使用双通道采样从图像序列生成循环过渡视频。",
        },
        "api_json": "wan_fflf_auto_v2_api.json",
        "inputs": [
            {"key": "images", "type": "gallery_select", "node_id": "278", "field": "inputs.folder_path",
             "required": True,
             "label": {"en": "Input Images", "ko": "입력 이미지", "zh": "输入图片"},
             "description": {"en": "Select images from gallery", "ko": "갤러리에서 이미지 선택", "zh": "从画廊选择图片"}},
            {"key": "positive_prompt", "type": "text", "node_id": "469", "field": "inputs.string_a",
             "default": "This scene is based on the 24fps, accelerated, fast-motion, Best quality, futuristic 8k, 4k UHD, Realistic delicacy, vibrant.\nA beautiful videoclip of a dynamic lovely photomodel.", "rows": 6,
             "label": {"en": "Master Prompt", "ko": "마스터 프롬프트", "zh": "主提示词"}},
            {"key": "negative_prompt", "type": "text", "node_id": "170", "field": "inputs.text",
             "default": "slow motion, gaudy, overexposed, blurred details, still image, low quality, cartoon, static", "rows": 4,
             "label": {"en": "Negative Prompt", "ko": "네거티브 프롬프트", "zh": "负面提示词"}},
            {"key": "segment_lengths", "type": "text", "node_id": "490", "field": "inputs.prompt",
             "default": "33\n37\n33\n33", "rows": 4,
             "label": {"en": "Segment Lengths (per line)", "ko": "세그먼트 길이 (줄당)", "zh": "段长度（每行）"}},
            {"key": "initial_width", "type": "number", "node_id": "501", "field": "inputs.value",
             "default": 288, "min": 64, "max": 2048, "step": 16,
             "label": {"en": "Initial Width", "ko": "초기 너비", "zh": "初始宽度"}},
            {"key": "upscale_factor", "type": "number", "node_id": "482", "field": "inputs.scale_factor",
             "default": 2.5, "min": 1.0, "max": 4.0, "step": 0.5,
             "label": {"en": "Upscale Factor", "ko": "업스케일 배율", "zh": "放大因子"}},
            {"key": "seed", "type": "number", "node_ids": ["408", "409"], "field": "inputs.noise_seed",
             "default": 138, "min": 0, "max": 2147483647, "step": 1,
             "label": {"en": "Seed", "ko": "시드", "zh": "种子"}},
            {"key": "looping", "type": "toggle", "node_id": "434", "field": "inputs.boolean",
             "default": True,
             "label": {"en": "Looping Video", "ko": "루프 비디오", "zh": "循环视频"}},
        ],
    },
}

# Multi-LoRA adapter configuration (Wan 2.2 High/Low Noise Mix strategy)
LORA_ADAPTERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora_adpts")
LORA_ADAPTERS = [
    {
        "name": "KoreanWoman",
        "category": "mov",
        "path": os.path.join(LORA_ADAPTERS_DIR, "mov", "character", "KoreanWoman.safetensors"),
        "type": "character",
        "default_high_weight": 0.3,
        "default_low_weight": 0.85,
        "alpha": 4.0,
        "rank": 32,
        "description": "Korean woman character LoRA (Wan2.2, Seoullina v2.1). Generates Korean female characters with improved skin texture and natural physics.",
        "trigger_words": ["a korean woman"],
        "semantic_tags": ["korean", "woman", "female", "asian", "natural", "realistic", "skin", "idol", "kpop"],
        "civitai_url": "https://civitai.com/models/1837542/wan22-korean-women",
        "preview_urls": [
            "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/3f1d4975-1614-43fb-a406-39dd194ef8e3/original=true/98613343.mp4",
            "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/9b43b67f-3a59-46b5-a1a7-45a1cb1c35c3/original=true/98613349.mp4",
        ],
    },
    {
        "name": "UlzzangG1",
        "category": "mov",
        "path": os.path.join(LORA_ADAPTERS_DIR, "mov", "character", "UlzzangG1.safetensors"),
        "type": "character",
        "default_high_weight": 0.0,
        "default_low_weight": 0.85,
        "alpha": 1.0,
        "rank": 32,
        "description": "Beautiful Korean Ulzzang Girl (Wan 2.2 T2V-A14B). Creates consistent character with large eyes, pale skin, makeup and small full lips. Low noise only (0.7-1.0).",
        "trigger_words": ["UlzzangG1", "a beautiful young korean woman with large eyes, pale skin, makeup and small full lips"],
        "semantic_tags": ["korean", "ulzzang", "girl", "cute", "pale", "makeup", "beauty", "idol"],
        "civitai_url": "https://civitai.com/models/2193272/beautiful-korean-ulzzang-girls-wan-22",
        "preview_urls": [],
    },
    {
        "name": "UkaSexyLight",
        "category": "mov",
        "path": os.path.join(LORA_ADAPTERS_DIR, "mov", "move", "wan22-uka-sexy-light.safetensors"),
        "type": "motion",
        "default_high_weight": 1.0,
        "default_low_weight": 0.0,
        "alpha": 1.0,
        "rank": 32,
        "description": "Uka Sexy Light motion LoRA (Wan 2.2 T2V-A14B). Generates realistic video with bright colors and a slight blur, like a dream. Strength 0.5-1.8 recommended.",
        "trigger_words": ["uka"],
        "semantic_tags": ["sexy", "light", "dreamy", "blur", "warm", "sensual", "dance", "glow"],
        "civitai_url": "https://civitai.com/models/1904858/wan22-uka-sexy-light",
        "preview_urls": [],
    },
    {
        "name": "HipSway",
        "category": "mov",
        "path": os.path.join(LORA_ADAPTERS_DIR, "mov", "move", "wan22_i2v_zxtp_hip_sway_low_r1.safetensors"),
        "type": "motion",
        "default_high_weight": 1.0,
        "default_low_weight": 0.0,
        "alpha": 32.0,
        "rank": 32,
        "description": "Hip sway motion LoRA (Wan 2.2 I2V-A14B). Generates hip swaying side-to-side motion. Trained by ZXTOPOWER. Strength 1.0 recommended. Lower quantization (Q4) may reduce motion quality — try lowering Shift to 5.",
        "trigger_words": ["She sways her hips side to side with her arms crossed"],
        "semantic_tags": ["hip", "sway", "dance", "motion", "sexy", "body", "movement"],
        "civitai_url": "https://civitai.com/models/2371603/wan-22-hip-sway",
        "preview_urls": [],
    },
    {
        "name": "OrbitCamV2",
        "category": "mov",
        "path": os.path.join(LORA_ADAPTERS_DIR, "mov", "camera", "wan2.2orbitcamv2_high_noise.safetensors"),
        "type": "camera",
        "default_high_weight": 0.8,
        "default_low_weight": 0.0,
        "alpha": 4.0,
        "rank": 4,
        "description": "Orbit camera LoRA (Wan 2.2 I2V). Makes the camera orbit around the subject. High-noise only — apply during early diffusion steps for camera motion control.",
        "trigger_words": ["ORBITCAM", "the viewer orbit around the subject"],
        "semantic_tags": ["camera", "orbit", "rotate", "360", "presentation", "product", "showcase"],
        "civitai_url": "https://civitai.com/models/2093287/luisap-wan-22-i2v-orbit-around-subject-v2",
        "preview_urls": [
            "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/6b1f0e1e-a6ed-4b8e-a0f7-5cb2e9f58c24/original=true/ComfyUI_00165_.mp4",
        ],
    },
]



def _scan_lora_dir(base_dir, category):
    """Scan a category directory for LoRA adapter files not already in LORA_ADAPTERS."""
    known_paths = {a["path"] for a in LORA_ADAPTERS}
    found = []
    cat_dir = os.path.join(base_dir, category)
    if not os.path.isdir(cat_dir):
        return found
    for sub_type in os.listdir(cat_dir):
        sub_path = os.path.join(cat_dir, sub_type)
        if not os.path.isdir(sub_path):
            continue
        for fname in os.listdir(sub_path):
            if fname.endswith(".safetensors"):
                full_path = os.path.join(sub_path, fname)
                if full_path not in known_paths:
                    found.append({
                        "name": os.path.splitext(fname)[0],
                        "category": category,
                        "path": full_path,
                        "type": sub_type,
                        "default_high_weight": 0.5,
                        "default_low_weight": 0.5,
                        "description": f"Auto-discovered {category}/{sub_type} LoRA",
                        "trigger_words": [],
                        "civitai_url": "",
                        "preview_urls": [],
                    })
    return found

# Auto-discover any LoRA adapters not explicitly listed
LORA_ADAPTERS.extend(_scan_lora_dir(LORA_ADAPTERS_DIR, "img"))
LORA_ADAPTERS.extend(_scan_lora_dir(LORA_ADAPTERS_DIR, "mov"))

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
flux_pipeline = None  # FLUX pipeline
upsampler = None      # Real-ESRGAN upsampler
active_model = None   # 's2v', 'i2v', or 'flux' — last-loaded model for API
models_on_gpu = set() # actual GPU occupancy: I2V on cuda:0, FLUX on cuda:1, S2V on both
generation_status = {}
gpu0_lock = threading.Lock()  # I2V (cuda:0) / S2V (both GPUs)
gpu1_lock = threading.Lock()  # FLUX (cuda:1) / S2V (both GPUs)
executor = ThreadPoolExecutor(max_workers=2)  # Allow concurrent I2V + FLUX

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
app.mount("/background", StaticFiles(directory="background"), name="background")

AVATARS_DIR = Path("settings/avatars")
AVATARS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/avatars", StaticFiles(directory=str(AVATARS_DIR)), name="avatars")

# Serve React frontend (built static files)
FRONTEND_DIR = Path(__file__).parent / "frontend" / "dist"
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")


# ============================================================================
# Database & Auth
# ============================================================================

async def init_db():
    """Create tables and seed super admin."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                name TEXT,
                picture TEXT,
                password_hash TEXT,
                role TEXT DEFAULT 'user',
                status TEXT DEFAULT 'pending',
                auth_provider TEXT DEFAULT 'google',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                file_type TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                original_name TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_uf_user ON user_files(user_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_uf_type ON user_files(user_id, file_type)")
        await db.commit()

        # Seed super admin
        if SUPER_ADMIN_EMAIL:
            cursor = await db.execute("SELECT id FROM users WHERE email = ?", (SUPER_ADMIN_EMAIL,))
            if not await cursor.fetchone():
                pw_hash = _bcrypt.hashpw(ADMIN_PASSWORD.encode('utf-8'), _bcrypt.gensalt()).decode('utf-8') if ADMIN_PASSWORD else None
                await db.execute(
                    "INSERT INTO users (email, name, password_hash, role, status, auth_provider) VALUES (?,?,?,?,?,?)",
                    (SUPER_ADMIN_EMAIL, "Super Admin", pw_hash, "superadmin", "approved", "local"),
                )
                await db.commit()
                logging.info(f"Created super admin: {SUPER_ADMIN_EMAIL}")


async def migrate_existing_files():
    """On first run, assign all existing files to the super admin account."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM user_files")
        row = await cursor.fetchone()
        if row[0] > 0:
            return
        cursor = await db.execute("SELECT id FROM users WHERE email = ?", (SUPER_ADMIN_EMAIL,))
        admin = await cursor.fetchone()
        if not admin:
            return
        admin_id = admin[0]
        IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
        AUD_EXT = {".wav", ".mp3", ".ogg", ".flac"}
        VID_EXT = {".mp4", ".avi", ".mov", ".webm"}

        for f in UPLOAD_DIR.iterdir():
            if f.is_file():
                ext = f.suffix.lower()
                ft = "upload_image" if ext in IMG_EXT else "upload_audio" if ext in AUD_EXT else "upload_video" if ext in VID_EXT else None
                if ft:
                    await db.execute(
                        "INSERT INTO user_files (user_id, file_type, filename, file_path, original_name) VALUES (?,?,?,?,?)",
                        (admin_id, ft, f.name, str(f), f.name),
                    )
        for f in OUTPUT_DIR.iterdir():
            if f.is_file() and f.suffix.lower() in (IMG_EXT | VID_EXT):
                await db.execute(
                    "INSERT INTO user_files (user_id, file_type, filename, file_path, original_name) VALUES (?,?,?,?,?)",
                    (admin_id, "output", f.name, str(f), f.name),
                )
        for f in BACKGROUNDS_DIR.iterdir():
            if f.is_file() and f.suffix.lower() in IMG_EXT:
                await db.execute(
                    "INSERT INTO user_files (user_id, file_type, filename, file_path, original_name) VALUES (?,?,?,?,?)",
                    (admin_id, "background", f.name, str(f), f.name),
                )
        # Migrate avatar files from settings/avatars/<group>/
        if AVATARS_DIR.exists():
            for group_dir in sorted(AVATARS_DIR.iterdir()):
                if group_dir.is_dir():
                    for f in sorted(group_dir.iterdir()):
                        if f.is_file() and f.suffix.lower() in IMG_EXT:
                            meta = _json.dumps({"group": group_dir.name})
                            await db.execute(
                                "INSERT INTO user_files (user_id, file_type, filename, file_path, original_name, metadata) VALUES (?,?,?,?,?,?)",
                                (admin_id, "avatar", f.name, str(f), f.name, meta),
                            )
        await db.commit()
        logging.info("Migrated existing files to super admin account")


@app.on_event("startup")
async def on_startup():
    await init_db()
    await migrate_existing_files()


# --- JWT helpers ---

def create_jwt_token(user_id: int, email: str, role: str) -> str:
    return pyjwt.encode(
        {"sub": str(user_id), "email": email, "role": role,
         "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRY_HOURS)},
        JWT_SECRET, algorithm=JWT_ALGORITHM,
    )


def decode_jwt_token(token: str) -> dict:
    try:
        payload = pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        payload["sub"] = int(payload["sub"])
        return payload
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except pyjwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")


# --- Auth dependencies ---

async def get_current_user(request: Request):
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Not authenticated")
    payload = decode_jwt_token(auth_header[7:])
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM users WHERE id = ?", (payload["sub"],))
        row = await cursor.fetchone()
    if not row:
        raise HTTPException(401, "User not found")
    user = dict(row)
    if user["status"] == "pending":
        raise HTTPException(403, "Account pending approval")
    if user["status"] == "suspended":
        raise HTTPException(403, "Account suspended")
    return user


async def get_admin_user(user=Depends(get_current_user)):
    if user["role"] != "superadmin":
        raise HTTPException(403, "Admin access required")
    return user


async def record_user_file(user_id: int, file_type: str, filename: str, file_path: str, original_name: str = None, metadata: str = None):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute(
            "INSERT INTO user_files (user_id, file_type, filename, file_path, original_name, metadata) VALUES (?,?,?,?,?,?)",
            (user_id, file_type, filename, file_path, original_name, metadata),
        )
        await db.commit()


# ============================================================================
# Request Models
# ============================================================================
class LoraWeightConfig(BaseModel):
    name: str
    high_weight: float = 0.0
    low_weight: float = 0.0


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
    lora_weights: Optional[list] = None  # List of {name, high_weight, low_weight}


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
    lora_weights: Optional[list] = None  # List of {name, high_weight, low_weight}


class FluxGenerateRequest(BaseModel):
    prompt: str = "K-pop idol, young Korean female, symmetrical face, V-shaped jawline, clear glass skin, double eyelids, trendy idol makeup.\nStage lighting, cinematic bokeh, pink and purple neon highlights, professional studio portrait, high-end fashion editorial style.\n8k resolution, photorealistic, raw photo, masterwork, intricate details of eyes and hair."
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    seed: int = -1
    aspect_ratio: str = "portrait"  # "portrait" (720x1280), "landscape" (1280x720), "square" (1024x1024)
    upscale: bool = False
    lora_weights: Optional[list] = None  # List of {name, weight}


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
    global pipeline, active_model, models_on_gpu
    if pipeline is not None:
        if 's2v' not in models_on_gpu:
            pipeline.noise_model.to(f'cuda:{LOCAL_RANK}')
            pipeline.text_encoder.model.to(f'cuda:{LOCAL_RANK}')
        active_model = 's2v'
        models_on_gpu.add('s2v')
        return pipeline

    logging.info(f"Loading Wan2.2-S2V-14B model (rank={LOCAL_RANK}, SP={USE_SP})...")
    cfg = WAN_CONFIGS['s2v-14B']

    # Build multi-LoRA config from available mov-category adapters
    multi_lora = [
        {"name": a["name"], "path": a["path"]}
        for a in LORA_ADAPTERS
        if os.path.exists(a["path"]) and a.get("category") == "mov"
    ]
    if multi_lora:
        logging.info(f"Multi-LoRA adapters found: {[a['name'] for a in multi_lora]}")
    else:
        logging.info("No multi-LoRA adapters found, checking legacy single LoRA...")

    # Fallback to legacy single LoRA if no multi-LoRA adapters
    lora_dir = None
    if not multi_lora and LORA_STRENGTH > 0 and os.path.exists(LORA_CHECKPOINT_DIR):
        lora_dir = LORA_CHECKPOINT_DIR

    if USE_SP:
        pipeline = wan.WanS2V(
            config=cfg,
            checkpoint_dir=CHECKPOINT_DIR,
            lora_checkpoint_dir=lora_dir,
            lora_strength=LORA_STRENGTH,
            multi_lora_configs=multi_lora if multi_lora else None,
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
            multi_lora_configs=multi_lora if multi_lora else None,
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
    models_on_gpu.add('s2v')
    logging.info(f"S2V model loaded successfully on rank {LOCAL_RANK}!")
    return pipeline




def load_i2v_pipeline():
    """Load Wan2.2 I2V pipeline with GGUF Q4_K_M quantized transformers (diffusers)."""
    global i2v_pipeline, active_model, models_on_gpu
    if i2v_pipeline is not None:
        if 'i2v' not in models_on_gpu:
            i2v_pipeline.to(f"cuda:{LOCAL_RANK}")
            models_on_gpu.add('i2v')
        active_model = 'i2v'
        return i2v_pipeline

    from diffusers import WanImageToVideoPipeline, WanTransformer3DModel, GGUFQuantizationConfig

    logging.info("Loading I2V GGUF Q4_K_M transformers (HighNoise + LowNoise)...")
    quantization_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)

    transformer_high = WanTransformer3DModel.from_single_file(
        I2V_GGUF_HIGH,
        config=I2V_DIFFUSERS_DIR,
        subfolder="transformer",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )
    transformer_low = WanTransformer3DModel.from_single_file(
        I2V_GGUF_LOW,
        config=I2V_DIFFUSERS_DIR,
        subfolder="transformer_2",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )

    logging.info("Loading I2V diffusers pipeline (T5, CLIP, VAE, scheduler)...")
    i2v_pipeline = WanImageToVideoPipeline.from_pretrained(
        I2V_DIFFUSERS_DIR,
        transformer=transformer_high,
        transformer_2=transformer_low,
        torch_dtype=torch.bfloat16,
    )

    # Load LoRA adapters onto each expert transformer
    mov_adapters = [a for a in LORA_ADAPTERS if a.get("category") == "mov" and os.path.exists(a["path"])]
    for adapter in mov_adapters:
        name = adapter["name"]
        hw = adapter.get("default_high_weight", 0.0)
        lw = adapter.get("default_low_weight", 0.0)
        try:
            if hw > 0:
                i2v_pipeline.load_lora_weights(adapter["path"], adapter_name=f"{name}_high")
                logging.info(f"LoRA '{name}_high' loaded onto transformer (scale={hw})")
            if lw > 0:
                i2v_pipeline.load_lora_weights(
                    adapter["path"], adapter_name=f"{name}_low",
                    load_into_transformer_2=True,
                )
                logging.info(f"LoRA '{name}_low' loaded onto transformer_2 (scale={lw})")
        except Exception as e:
            logging.warning(f"Failed to load LoRA '{name}': {e}")

    # Load all components onto single GPU
    i2v_pipeline.to(f"cuda:{LOCAL_RANK}")

    active_model = 'i2v'
    models_on_gpu.add('i2v')
    logging.info(f"I2V GGUF pipeline loaded on cuda:{LOCAL_RANK}!")
    return i2v_pipeline


def load_flux_pipeline():
    """Load FLUX.2-klein-9B pipeline for image generation."""
    global flux_pipeline, active_model, models_on_gpu
    flux_gpu = 1 if torch.cuda.device_count() >= 2 else 0

    if flux_pipeline is not None:
        if 'flux' not in models_on_gpu:
            flux_pipeline.to(f"cuda:{flux_gpu}")
            models_on_gpu.add('flux')
        active_model = 'flux'
        return flux_pipeline

    from diffusers import Flux2KleinPipeline

    logging.info(f"Loading FLUX.2-klein-9B model on cuda:{flux_gpu}...")
    flux_pipeline = Flux2KleinPipeline.from_pretrained(
        FLUX_MODEL_ID,
        torch_dtype=torch.bfloat16,
        cache_dir=FLUX_CACHE_DIR,
    )
    flux_pipeline.to(f'cuda:{flux_gpu}')

    active_model = 'flux'
    models_on_gpu.add('flux')
    logging.info(f"FLUX.2-klein-9B model loaded on cuda:{flux_gpu}!")
    return flux_pipeline


def get_upsampler():
    """Get or initialize Real-ESRGAN x2 upsampler."""
    global upsampler
    if upsampler is not None:
        return upsampler

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    logging.info("Initializing Real-ESRGAN x2 upsampler...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    upsampler = RealESRGANer(
        scale=2,
        model_path=REALESRGAN_MODEL_PATH,
        model=model,
        tile=512,
        half=True,
    )
    logging.info("Real-ESRGAN upsampler ready!")
    return upsampler


def _offload_s2v():
    """Offload S2V model from both GPUs to CPU."""
    global models_on_gpu
    if 's2v' in models_on_gpu and pipeline is not None:
        logging.info("Moving S2V model to CPU...")
        pipeline.noise_model.cpu()
        pipeline.text_encoder.model.cpu()
        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
            pipeline.vae.model.cpu()
        if hasattr(pipeline, 'audio') and pipeline.audio is not None:
            pipeline.audio.cpu()
        models_on_gpu.discard('s2v')


def _offload_i2v():
    """Offload I2V pipeline from cuda:0 to CPU."""
    global models_on_gpu
    if 'i2v' in models_on_gpu and i2v_pipeline is not None:
        logging.info("Moving I2V pipeline to CPU...")
        i2v_pipeline.to("cpu")
        models_on_gpu.discard('i2v')


def _offload_flux():
    """Offload FLUX pipeline from cuda:1 to CPU."""
    global flux_pipeline, models_on_gpu
    if 'flux' in models_on_gpu and flux_pipeline is not None:
        logging.info("Unloading FLUX model from GPU...")
        for attr in ('transformer', 'text_encoder', 'text_encoder_2', 'vae', 'scheduler'):
            if hasattr(flux_pipeline, attr):
                delattr(flux_pipeline, attr)
        flux_pipeline = None
        models_on_gpu.discard('flux')


def ensure_model_loaded(model_type: str):
    """Ensure the requested model is on GPU.

    GPU layout:
      - I2V: cuda:0 (~20 GB GGUF)
      - FLUX: cuda:1 (~18 GB)
      - S2V: both GPUs (Sequence Parallel)

    I2V and FLUX can coexist on separate GPUs.
    S2V requires exclusive access to both GPUs.
    """
    global active_model, models_on_gpu

    if model_type in models_on_gpu:
        active_model = model_type
        return

    logging.info(f"Loading model: {model_type} (currently on GPU: {models_on_gpu})")

    if model_type == 's2v':
        # S2V needs both GPUs — offload everything
        _offload_i2v()
        _offload_flux()
        gc.collect()
        torch.cuda.empty_cache()
        load_pipeline()
    elif model_type == 'i2v':
        # I2V uses cuda:0 — only S2V conflicts
        _offload_s2v()
        gc.collect()
        torch.cuda.empty_cache()
        load_i2v_pipeline()
    elif model_type == 'flux':
        # FLUX uses cuda:1 — only S2V conflicts
        _offload_s2v()
        gc.collect()
        torch.cuda.empty_cache()
        load_flux_pipeline()

    logging.info(f"Model loaded: {model_type}. On GPU: {models_on_gpu}")


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
            logging.info(f"Rank {LOCAL_RANK}: Swapping to S2V model...")
            try:
                ensure_model_loaded('s2v')
                pipeline.generate(**params)
                logging.info(f"Rank {LOCAL_RANK}: S2V generation done.")
            except Exception as e:
                logging.error(f"Rank {LOCAL_RANK}: S2V generation error: {e}")
                import traceback
                traceback.print_exc()

        elif cmd.item() == 2:  # I2V Generate (diffusers pipeline runs on rank 0 only)
            params = broadcast_generate_params(None)
            logging.info(f"Rank {LOCAL_RANK}: I2V uses diffusers pipeline on rank 0 — skipping.")

        elif cmd.item() == -1:  # Shutdown
            logging.info(f"Rank {LOCAL_RANK}: Shutting down.")
            break


# ============================================================================
# Background Tasks
# ============================================================================
def generate_video_task(task_id: str, params: dict):
    """Background task for S2V video generation (runs in thread pool)."""
    global pipeline, generation_status

    # S2V uses both GPUs — acquire both locks
    if not gpu0_lock.acquire(blocking=False):
        generation_status[task_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Waiting for GPU 0 (another generation in progress)...",
            "output_path": None,
        }
        gpu0_lock.acquire()
    gpu1_lock.acquire()  # also need GPU 1

    try:
        logging.info(f"Starting S2V generation task: {task_id}")
        generation_status[task_id] = {
            "status": "processing",
            "progress": 0.02,
            "message": "Switching to S2V model...",
            "output_path": None,
        }

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

        # Build LoRA weights dict for pipeline
        lora_weights = None
        if params.get("lora_weights"):
            lora_weights = {}
            for lw in params["lora_weights"]:
                lora_weights[lw["name"]] = {
                    "high_weight": lw.get("high_weight", 0.0),
                    "low_weight": lw.get("low_weight", 0.0),
                }

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
            lora_weights=lora_weights,
        )

        # Signal SP workers FIRST so they can start model swap concurrently
        if USE_SP:
            cmd = torch.tensor([1], dtype=torch.long, device=f'cuda:{LOCAL_RANK}')
            dist.broadcast(cmd, src=0)
            broadcast_generate_params(gen_kwargs)

        # Swap model on rank 0 AFTER signaling workers (avoids barrier deadlock)
        ensure_model_loaded('s2v')

        generation_status[task_id]["progress"] = 0.1
        generation_status[task_id]["message"] = "Preparing generation..."

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

        _uid = params.get("_user_id")
        if _uid and final_path:
            import asyncio as _aio
            try:
                loop = _aio.new_event_loop()
                loop.run_until_complete(record_user_file(_uid, "output", os.path.basename(final_path), final_path, os.path.basename(final_path)))
                loop.close()
            except Exception:
                pass

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
        gpu1_lock.release()
        gpu0_lock.release()


def generate_i2v_task(task_id: str, params: dict):
    """Background task for I2V video generation (diffusers GGUF pipeline)."""
    global i2v_pipeline, generation_status

    # I2V uses cuda:0 only
    if not gpu0_lock.acquire(blocking=False):
        generation_status[task_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Waiting for GPU 0 (another generation in progress)...",
            "output_path": None,
        }
        gpu0_lock.acquire()

    try:
        logging.info(f"Starting I2V generation task: {task_id}")
        generation_status[task_id] = {
            "status": "processing",
            "progress": 0.02,
            "message": "Switching to I2V model...",
            "output_path": None,
        }

        seed = params["seed"]
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        # Load input image
        img = Image.open(params["image_path"]).convert("RGB")
        img_w, img_h = img.size  # PIL: (width, height)

        # Parse resolution, auto-correct to match input image aspect ratio
        try:
            res_parts = params["resolution"].split("*")
            res_height, res_width = int(res_parts[0]), int(res_parts[1])
        except Exception:
            res_height, res_width = 720, 1280

        # If image is portrait but resolution is landscape (or vice versa), swap
        img_is_portrait = img_h > img_w
        res_is_portrait = res_height > res_width
        if img_is_portrait != res_is_portrait:
            res_height, res_width = res_width, res_height
            logging.info(f"Resolution auto-corrected to match image orientation: {res_height}x{res_width}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.info(f"Starting I2V generation: {res_height}x{res_width}, steps={params['inference_steps']}, seed={seed}")

        # Signal SP workers (they'll just no-op for I2V now)
        if USE_SP:
            cmd = torch.tensor([2], dtype=torch.long, device=f'cuda:{LOCAL_RANK}')
            dist.broadcast(cmd, src=0)
            # Broadcast dummy params so workers don't hang
            broadcast_generate_params({"_noop": True})

        # Swap model — loads I2V diffusers pipeline if not already loaded
        ensure_model_loaded('i2v')

        # Configure LoRA adapter weights for this generation (after pipeline is loaded)
        # diffusers bakes alpha/rank scaling into weights during Kohya→PEFT conversion,
        # so user_weight=1.0 directly equals ComfyUI strength=1.0. No correction needed.
        # WanLoraLoaderMixin.set_adapters() auto-distributes across transformer + transformer_2.
        i2v_pipeline.disable_lora()  # Disable all adapters first
        if params.get("lora_weights"):
            all_adapter_names, all_adapter_weights = [], []
            for lw in params["lora_weights"]:
                name = lw["name"]
                hw = lw.get("high_weight", 0.0)
                lwt = lw.get("low_weight", 0.0)
                if hw > 0:
                    all_adapter_names.append(f"{name}_high")
                    all_adapter_weights.append(hw)
                    logging.info(f"LoRA {name} HIGH: weight={hw}")
                if lwt > 0:
                    all_adapter_names.append(f"{name}_low")
                    all_adapter_weights.append(lwt)
                    logging.info(f"LoRA {name} LOW: weight={lwt}")
            if all_adapter_names:
                try:
                    i2v_pipeline.enable_lora()
                    i2v_pipeline.set_adapters(all_adapter_names, adapter_weights=all_adapter_weights)
                    logging.info(f"LoRA adapters activated: {list(zip(all_adapter_names, all_adapter_weights))}")
                except Exception as e:
                    logging.warning(f"Failed to set LoRA adapters: {e}")
                    import traceback
                    traceback.print_exc()

        generation_status[task_id]["progress"] = 0.1
        generation_status[task_id]["message"] = "Generating video with GGUF Q4_K_M..."

        # Progress callback for diffusers pipeline
        total_steps = params["inference_steps"]

        def step_callback(pipe, step, timestep, callback_kwargs):
            progress = 0.1 + 0.8 * ((step + 1) / total_steps)
            generation_status[task_id]["progress"] = round(progress, 3)
            generation_status[task_id]["message"] = f"Denoising step {step + 1}/{total_steps}..."
            return callback_kwargs

        generator = torch.Generator(device="cpu").manual_seed(seed)

        output = i2v_pipeline(
            image=img,
            prompt=params["prompt"],
            negative_prompt=params["negative_prompt"],
            height=res_height,
            width=res_width,
            num_frames=params["frame_num"],
            num_inference_steps=params["inference_steps"],
            guidance_scale=params["guidance_scale"],
            generator=generator,
            callback_on_step_end=step_callback,
        )

        generation_status[task_id]["progress"] = 0.9
        generation_status[task_id]["message"] = "Saving video..."

        # Save video from diffusers output (list of PIL Images)
        from diffusers.utils import export_to_video
        video_path = str(OUTPUT_DIR / f"i2v_{timestamp}.mp4")
        export_to_video(output.frames[0], video_path, fps=16)

        final_path = f"/outputs/i2v_{timestamp}.mp4"

        del output
        gc.collect()
        torch.cuda.empty_cache()

        _uid = params.get("_user_id")
        if _uid and final_path:
            import asyncio as _aio
            try:
                loop = _aio.new_event_loop()
                loop.run_until_complete(record_user_file(_uid, "output", os.path.basename(final_path), final_path, os.path.basename(final_path)))
                loop.close()
            except Exception:
                pass

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
        gpu0_lock.release()


def generate_flux_task(task_id: str, params: dict):
    """Background task for FLUX image generation."""
    global flux_pipeline, generation_status

    # FLUX uses cuda:1 only
    if not gpu1_lock.acquire(blocking=False):
        generation_status[task_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Waiting for GPU 1 (another generation in progress)...",
            "output_path": None,
        }
        gpu1_lock.acquire()

    try:
        logging.info(f"Starting FLUX generation task: {task_id}")
        generation_status[task_id] = {
            "status": "processing",
            "progress": 0.02,
            "message": "Switching to FLUX model...",
            "output_path": None,
        }

        ensure_model_loaded('flux')

        generation_status[task_id]["progress"] = 0.1
        generation_status[task_id]["message"] = "Preparing FLUX generation..."

        seed = params["seed"]
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine image dimensions from aspect ratio
        ar = params.get("aspect_ratio", "portrait")
        if ar == "landscape":
            gen_height, gen_width = 720, 1280
        elif ar == "square":
            gen_height, gen_width = 1024, 1024
        else:  # portrait
            gen_height, gen_width = 1280, 720

        logging.info(f"Starting FLUX.2-klein-9B generation: seed={seed}, steps={params['num_inference_steps']}, {gen_width}x{gen_height} ({ar})")

        generation_status[task_id]["progress"] = 0.15
        generation_status[task_id]["message"] = "Generating image..."

        flux_gpu = 1 if torch.cuda.device_count() >= 2 else 0
        generator = torch.Generator(device=f'cuda:{flux_gpu}').manual_seed(seed)
        image = flux_pipeline(
            prompt=params["prompt"],
            height=gen_height,
            width=gen_width,
            guidance_scale=params["guidance_scale"],
            num_inference_steps=params["num_inference_steps"],
            generator=generator,
        ).images[0]

        generation_status[task_id]["progress"] = 0.7
        generation_status[task_id]["message"] = "Saving image..."

        # Save original image
        image_path = str(OUTPUT_DIR / f"flux_{timestamp}.png")
        image.save(image_path)
        final_path = f"/outputs/flux_{timestamp}.png"
        upscaled_path = None

        # Upscale if requested
        if params.get("upscale", True) and os.path.exists(REALESRGAN_MODEL_PATH):
            generation_status[task_id]["progress"] = 0.75
            generation_status[task_id]["message"] = "Upscaling with Real-ESRGAN x2..."
            try:
                import cv2
                import numpy as np
                cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                esrgan = get_upsampler()
                output, _ = esrgan.enhance(cv2_image, outscale=2)
                upscaled_filename = f"flux_{timestamp}_x2.png"
                upscaled_file_path = str(OUTPUT_DIR / upscaled_filename)
                cv2.imwrite(upscaled_file_path, output)
                upscaled_path = f"/outputs/{upscaled_filename}"
                logging.info(f"Upscaled image saved: {upscaled_file_path}")
            except Exception as up_err:
                logging.warning(f"Upscaling failed: {up_err}")

        # Cleanup
        del image
        gc.collect()
        torch.cuda.empty_cache()

        _uid = params.get("_user_id")
        if _uid and image_path:
            import asyncio as _aio
            try:
                loop = _aio.new_event_loop()
                loop.run_until_complete(record_user_file(_uid, "output", os.path.basename(image_path), image_path, os.path.basename(image_path)))
                loop.close()
            except Exception:
                pass

        generation_status[task_id] = {
            "status": "completed",
            "progress": 1.0,
            "message": "FLUX generation completed!",
            "output_path": final_path,
            "absolute_path": image_path,
            "upscaled_path": upscaled_path,
            "seed": seed,
        }

    except Exception as e:
        logging.error(f"FLUX generation failed: {e}")
        import traceback
        traceback.print_exc()
        generation_status[task_id] = {
            "status": "failed",
            "progress": 0,
            "message": str(e),
            "output_path": None,
        }
    finally:
        gpu1_lock.release()


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
        "flux_model_loaded": flux_pipeline is not None,
        "active_model": active_model,
        "lora_enabled": LORA_STRENGTH > 0 and os.path.exists(LORA_CHECKPOINT_DIR),
        "lora_strength": LORA_STRENGTH,
        "gpu0_busy": gpu0_lock.locked(),
        "gpu1_busy": gpu1_lock.locked(),
        "sequence_parallel": USE_SP,
        "world_size": WORLD_SIZE,
        "i2v_available": os.path.exists(I2V_GGUF_HIGH) and os.path.exists(I2V_GGUF_LOW),
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
        "i2v_available": os.path.exists(I2V_GGUF_HIGH) and os.path.exists(I2V_GGUF_LOW),
        "i2v_default_steps": 40,
        "i2v_default_guidance": 5.0,
        "i2v_default_frame_num": 81,
        "i2v_default_shift": 5.0,
        # LoRA info
        "lora_adapters_available": any(os.path.exists(a["path"]) for a in LORA_ADAPTERS),
        # Auth
        "google_client_id": GOOGLE_CLIENT_ID,
    }


@app.get("/api/lora-adapters")
async def get_lora_adapters(category: Optional[str] = None, user=Depends(get_current_user)):
    """List available LoRA adapters with metadata, optionally filtered by category."""
    adapters = []
    for a in LORA_ADAPTERS:
        if category and a.get("category") != category:
            continue
        exists = os.path.exists(a["path"])
        adapters.append({
            "name": a["name"],
            "category": a.get("category", "mov"),
            "type": a["type"],
            "available": exists,
            "size_mb": round(os.path.getsize(a["path"]) / 1024 / 1024, 1) if exists else 0,
            "default_high_weight": a["default_high_weight"],
            "default_low_weight": a["default_low_weight"],
            "description": a.get("description", ""),
            "trigger_words": a.get("trigger_words", []),
            "civitai_url": a.get("civitai_url", ""),
            "preview_urls": a.get("preview_urls", []),
        })
    return {"adapters": adapters}


# ============================================================================
# Auth Endpoints
# ============================================================================

class LoginRequest(BaseModel):
    email: str
    password: str


class GoogleAuthRequest(BaseModel):
    credential: str


@app.post("/api/auth/login")
async def auth_login(req: LoginRequest):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM users WHERE email = ? AND auth_provider = 'local'", (req.email,))
        row = await cursor.fetchone()
    if not row:
        raise HTTPException(401, "Invalid credentials")
    user = dict(row)
    if not user["password_hash"] or not _bcrypt.checkpw(req.password.encode('utf-8'), user["password_hash"].encode('utf-8')):
        raise HTTPException(401, "Invalid credentials")
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user["id"],))
        await db.commit()
    token = create_jwt_token(user["id"], user["email"], user["role"])
    return {"token": token, "user": {"id": user["id"], "email": user["email"], "name": user["name"],
            "role": user["role"], "status": user["status"], "picture": user.get("picture")}}


@app.post("/api/auth/google")
async def auth_google(req: GoogleAuthRequest):
    from google.oauth2 import id_token
    from google.auth.transport import requests as google_requests
    try:
        idinfo = id_token.verify_oauth2_token(req.credential, google_requests.Request(), GOOGLE_CLIENT_ID)
    except ValueError:
        raise HTTPException(401, "Invalid Google token")
    email = idinfo.get("email")
    name = idinfo.get("name", "")
    picture = idinfo.get("picture", "")
    if not email:
        raise HTTPException(400, "No email in Google token")

    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM users WHERE email = ?", (email,))
        row = await cursor.fetchone()
        if row:
            user = dict(row)
            await db.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP, name = ?, picture = ? WHERE id = ?",
                             (name, picture, user["id"]))
            await db.commit()
        else:
            status = "approved" if email == SUPER_ADMIN_EMAIL else "pending"
            role = "superadmin" if email == SUPER_ADMIN_EMAIL else "user"
            cursor = await db.execute(
                "INSERT INTO users (email, name, picture, role, status, auth_provider, last_login) VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)",
                (email, name, picture, role, status, "google"),
            )
            await db.commit()
            user = {"id": cursor.lastrowid, "email": email, "name": name, "picture": picture,
                    "role": role, "status": status}

    token = create_jwt_token(user["id"], user["email"], user["role"])
    return {"token": token, "user": {"id": user["id"], "email": user["email"], "name": user["name"],
            "role": user["role"], "status": user["status"], "picture": user.get("picture")}}


@app.get("/api/auth/me")
async def auth_me(user=Depends(get_current_user)):
    return {"user": {"id": user["id"], "email": user["email"], "name": user["name"],
            "role": user["role"], "status": user["status"], "picture": user.get("picture")}}


# ============================================================================
# Admin Endpoints
# ============================================================================

@app.get("/api/admin/users")
async def admin_list_users(user=Depends(get_admin_user)):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, email, name, picture, role, status, auth_provider, created_at, last_login FROM users ORDER BY created_at DESC")
        users = [dict(r) for r in await cursor.fetchall()]
    return {"users": users}


@app.post("/api/admin/users/{user_id}/approve")
async def admin_approve_user(user_id: int, user=Depends(get_admin_user)):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("UPDATE users SET status = 'approved' WHERE id = ? AND status = 'pending'", (user_id,))
        await db.commit()
    return {"message": "User approved"}


@app.post("/api/admin/users/{user_id}/suspend")
async def admin_suspend_user(user_id: int, user=Depends(get_admin_user)):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("UPDATE users SET status = 'suspended' WHERE id = ? AND id != ?", (user_id, user["id"]))
        await db.commit()
    return {"message": "User suspended"}


@app.post("/api/admin/users/{user_id}/activate")
async def admin_activate_user(user_id: int, user=Depends(get_admin_user)):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("UPDATE users SET status = 'approved' WHERE id = ? AND status = 'suspended'", (user_id,))
        await db.commit()
    return {"message": "User activated"}


@app.delete("/api/admin/users/{user_id}")
async def admin_delete_user(user_id: int, user=Depends(get_admin_user)):
    if user_id == user["id"]:
        raise HTTPException(400, "Cannot delete yourself")
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("DELETE FROM user_files WHERE user_id = ?", (user_id,))
        await db.execute("DELETE FROM users WHERE id = ?", (user_id,))
        await db.commit()
    return {"message": "User deleted"}


# ============================================================================
# Upload / File Endpoints
# ============================================================================

@app.post("/api/upload/image")
async def upload_image(file: UploadFile = File(...), user=Depends(get_current_user)):
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

        await record_user_file(user["id"], "upload_image", filename, str(filepath),
                               file.filename, _json.dumps({"width": width, "height": height}))

        return {
            "path": str(filepath),
            "url": f"/uploads/{filename}",
            "width": width,
            "height": height,
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error("Image upload failed: %s", e, exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/api/upload/audio")
async def upload_audio(file: UploadFile = File(...), user=Depends(get_current_user)):
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

        await record_user_file(user["id"], "upload_audio", filename, str(filepath), file.filename)

        return {"path": str(filepath), "url": f"/uploads/{filename}"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/upload/video")
async def upload_video(file: UploadFile = File(...), user=Depends(get_current_user)):
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

        await record_user_file(user["id"], "upload_video", filename, str(filepath), file.filename)

        return {"path": str(filepath), "url": f"/uploads/{filename}"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/upload/background")
async def upload_background(file: UploadFile = File(...), user=Depends(get_current_user)):
    """Upload background image and save to backgrounds/stages/."""
    try:
        ext = Path(file.filename).suffix.lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            raise HTTPException(400, "Invalid image format")

        # Use original filename (sanitized) to keep it recognizable
        safe_name = "".join(c for c in Path(file.filename).stem if c.isalnum() or c in "-_ ").strip()
        if not safe_name:
            safe_name = str(uuid.uuid4())
        filename = f"{safe_name}{ext}"
        # Avoid overwriting: append uuid if exists
        filepath = BACKGROUNDS_DIR / filename
        if filepath.exists():
            filename = f"{safe_name}_{uuid.uuid4().hex[:8]}{ext}"
            filepath = BACKGROUNDS_DIR / filename

        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)

        with Image.open(filepath) as img:
            width, height = img.size

        await record_user_file(user["id"], "background", filename, str(filepath),
                               file.filename, _json.dumps({"width": width, "height": height}))

        return {
            "path": str(filepath),
            "url": f"/background/stages/{filename}",
            "width": width,
            "height": height,
            "filename": filename,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/backgrounds")
async def list_backgrounds(user=Depends(get_current_user)):
    """List background images owned by current user."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM user_files WHERE user_id = ? AND file_type = 'background' ORDER BY created_at DESC",
            (user["id"],))
        rows = [dict(r) for r in await cursor.fetchall()]
    images = []
    for r in rows:
        fp = Path(r["file_path"])
        if fp.exists():
            meta = _json.loads(r["metadata"]) if r["metadata"] else {}
            images.append({
                "filename": r["filename"], "url": f"/background/stages/{r['filename']}",
                "path": r["file_path"], "width": meta.get("width", 0), "height": meta.get("height", 0),
            })
    return {"backgrounds": images, "total": len(images)}


@app.get("/api/avatars")
async def list_avatar_groups(user=Depends(get_current_user)):
    """List avatar groups owned by current user."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        cursor = await db.execute(
            "SELECT DISTINCT json_extract(metadata, '$.group') as grp FROM user_files WHERE user_id = ? AND file_type = 'avatar' AND metadata IS NOT NULL ORDER BY grp",
            (user["id"],))
        rows = await cursor.fetchall()
    groups = [r[0] for r in rows if r[0]]
    return {"groups": groups}


@app.get("/api/avatars/{group}")
async def list_avatar_images(group: str, user=Depends(get_current_user)):
    """List avatar images in a specific group owned by current user."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM user_files WHERE user_id = ? AND file_type = 'avatar' AND json_extract(metadata, '$.group') = ? ORDER BY filename",
            (user["id"], group))
        rows = [dict(r) for r in await cursor.fetchall()]
    images = []
    for r in rows:
        fp = Path(r["file_path"])
        if fp.exists():
            images.append({
                "filename": r["filename"],
                "url": f"/avatars/{group}/{r['filename']}",
                "path": r["file_path"],
            })
    return {"images": images, "group": group}


@app.delete("/api/avatars/{group}/{filename}")
async def delete_avatar_image(group: str, filename: str, user=Depends(get_current_user)):
    """Delete an avatar image from a group."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        cursor = await db.execute(
            "SELECT id, file_path FROM user_files WHERE user_id = ? AND file_type = 'avatar' AND filename = ? AND json_extract(metadata, '$.group') = ?",
            (user["id"], filename, group))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Avatar image not found")
        file_id, file_path = row
        fp = Path(file_path)
        if fp.exists():
            fp.unlink()
        await db.execute("DELETE FROM user_files WHERE id = ?", (file_id,))
        await db.commit()
    logging.info(f"Deleted avatar image: {group}/{filename} (user={user['id']})")
    return {"ok": True, "deleted": filename, "group": group}


@app.post("/api/register-avatar")
async def register_avatar(request: Request, user=Depends(get_current_user)):
    """Register a gallery/output image as an avatar in a specified group."""
    import shutil
    data = await request.json()
    source_path = data.get("source_path", "").strip()
    group = data.get("group", "").strip()
    if not source_path or not group:
        raise HTTPException(400, "source_path and group are required")

    # Sanitize group name
    group = "".join(c for c in group if c.isalnum() or c in "_-").strip()
    if not group:
        raise HTTPException(400, "Invalid group name")

    # Resolve source file
    src = Path(source_path)
    if not src.is_absolute():
        src = Path.cwd() / src
    if not src.exists():
        raise HTTPException(404, f"Source file not found: {source_path}")
    if src.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(400, "Invalid image format")

    # Copy to avatar directory
    avatar_dir = AVATARS_DIR / group
    avatar_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    avatar_fn = f"avatar_{ts}{src.suffix.lower()}"
    avatar_fp = avatar_dir / avatar_fn
    shutil.copy2(str(src), str(avatar_fp))

    # Register in DB
    await record_user_file(user["id"], "avatar", avatar_fn, str(avatar_fp),
                           src.name, _json.dumps({"group": group}))

    logging.info(f"Registered avatar: {group}/{avatar_fn} from {source_path} (user={user['id']})")
    return {
        "ok": True,
        "group": group,
        "filename": avatar_fn,
        "url": f"/avatars/{group}/{avatar_fn}",
        "path": str(avatar_fp),
    }


@app.post("/api/prepare-avatar")
async def prepare_avatar(request: Request, user=Depends(get_current_user)):
    """Run avatar prepare pipeline: pose edit + face swap, then register as avatar."""
    data = await request.json()
    source_path = data.get("source_path", "").strip()
    group = data.get("group", "").strip()
    if not source_path or not group:
        raise HTTPException(400, "source_path and group are required")
    group = "".join(c for c in group if c.isalnum() or c in "_-").strip()
    if not group:
        raise HTTPException(400, "Invalid group name")
    src = Path(source_path)
    if not src.is_absolute():
        src = Path.cwd() / src
    if not src.exists():
        raise HTTPException(404, f"Source file not found: {source_path}")

    task_id = str(uuid.uuid4())
    generation_status[task_id] = {
        "status": "processing", "progress": 0.0,
        "message": "Starting avatar preparation...", "output_path": None,
    }
    _user_id = user["id"]
    executor.submit(avatar_prepare_task, task_id, str(src), group, _user_id)
    return {"task_id": task_id}


def avatar_prepare_task(task_id: str, source_path: str, group: str, user_id: int):
    """Background task: 2-step avatar preparation pipeline."""
    try:
        gpu0_lock.acquire()
        gpu1_lock.acquire()

        # Offload models
        generation_status[task_id].update({"progress": 0.02, "message": "Offloading models..."})
        _offload_s2v()
        _offload_i2v()
        _offload_flux()
        gc.collect()
        torch.cuda.empty_cache()

        generation_status[task_id].update({"progress": 0.05, "message": "Starting ComfyUI..."})
        ensure_comfyui_running()

        # ── Step 1: Klein pose edit ──
        generation_status[task_id].update({"progress": 0.08, "message": "Step 1: Uploading image..."})
        source_comfyui = upload_to_comfyui(source_path)

        # Load pose edit workflow
        pose_wf_path = WORKFLOW_DIR / "flux_klein_pose_edit_api.json"
        with open(pose_wf_path) as f:
            pose_wf = json_mod.load(f)

        pose_wf["10"]["inputs"]["image"] = source_comfyui
        pose_wf["92"]["inputs"]["value"] = random.randint(0, 2**53)

        generation_status[task_id].update({"progress": 0.10, "message": "Step 1: Generating pose edit..."})
        client_id1 = str(uuid.uuid4())
        resp1 = http_requests.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": pose_wf, "client_id": client_id1},
        )
        if resp1.status_code != 200:
            raise Exception(f"ComfyUI rejected pose edit: {resp1.text}")
        prompt_id1 = resp1.json()["prompt_id"]

        try:
            monitor_comfyui_progress(task_id, prompt_id1, client_id1)
        except Exception as e1:
            logging.warning(f"Step 1 monitor timeout, polling: {e1}")
            import time as _time
            for _ in range(60):
                _time.sleep(10)
                try:
                    h = http_requests.get(f"{COMFYUI_URL}/history/{prompt_id1}").json()
                    if h.get(prompt_id1, {}).get("outputs", {}):
                        break
                except Exception:
                    pass
            else:
                raise Exception(f"Step 1 timed out: {e1}")

        generation_status[task_id].update({"progress": 0.45, "message": "Step 1: Retrieving result..."})
        step1_output = retrieve_comfyui_output(prompt_id1)
        step1_abs = str(OUTPUT_DIR / os.path.basename(step1_output))
        logging.info(f"Avatar prepare Step 1 complete: {step1_output}")

        # ── Step 2: BFS Face Swap to restore original face ──
        generation_status[task_id].update({"progress": 0.50, "message": "Step 2: Cropping face & uploading..."})
        step1_comfyui = upload_to_comfyui(step1_abs)
        # Crop original image to head-only so clothing doesn't leak into face swap
        cropped_face_path = crop_face_head(source_path)
        original_comfyui = upload_to_comfyui(cropped_face_path)
        # Clean up cropped temp file after upload
        if cropped_face_path != source_path and os.path.exists(cropped_face_path):
            try:
                os.remove(cropped_face_path)
            except OSError:
                pass

        faceswap_path = WORKFLOW_DIR / "flux_klein_faceswap_api.json"
        with open(faceswap_path) as f:
            faceswap_wf = json_mod.load(f)

        faceswap_wf["10"]["inputs"]["image"] = step1_comfyui     # target body (pose-edited)
        faceswap_wf["11"]["inputs"]["image"] = original_comfyui  # face source (original)
        faceswap_wf["92"]["inputs"]["value"] = random.randint(0, 2**53)
        faceswap_wf["21"]["inputs"]["strength_model"] = 0.85     # Lower LoRA for softer face restoration

        generation_status[task_id].update({"progress": 0.55, "message": "Step 2: Running face swap..."})
        client_id2 = str(uuid.uuid4())
        resp2 = http_requests.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": faceswap_wf, "client_id": client_id2},
        )
        if resp2.status_code != 200:
            raise Exception(f"ComfyUI rejected face swap: {resp2.text}")
        prompt_id2 = resp2.json()["prompt_id"]

        try:
            monitor_comfyui_progress(task_id, prompt_id2, client_id2)
        except Exception as e2:
            logging.warning(f"Step 2 monitor timeout, polling: {e2}")
            import time as _time2
            for _ in range(60):
                _time2.sleep(10)
                try:
                    h2 = http_requests.get(f"{COMFYUI_URL}/history/{prompt_id2}").json()
                    if h2.get(prompt_id2, {}).get("outputs", {}):
                        break
                except Exception:
                    pass
            else:
                raise Exception(f"Step 2 timed out: {e2}")

        generation_status[task_id].update({"progress": 0.90, "message": "Step 2: Retrieving result..."})
        final_output = retrieve_comfyui_output(prompt_id2)
        final_abs = str(OUTPUT_DIR / os.path.basename(final_output))
        logging.info(f"Avatar prepare Step 2 complete: {final_output}")

        # Composite swapped face onto original pose-edited body
        if os.path.exists(final_abs) and os.path.exists(step1_abs):
            generation_status[task_id].update({"progress": 0.92, "message": "Step 2: Compositing face..."})
            try:
                composite_face_onto_body(final_abs, step1_abs)
            except Exception as comp_err:
                logging.warning(f"Avatar prepare face composite failed: {comp_err}")

        # ── Step 3: Register as avatar ──
        generation_status[task_id].update({"progress": 0.95, "message": "Registering as avatar..."})
        import shutil
        avatar_dir = AVATARS_DIR / group
        avatar_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = os.path.splitext(final_abs)[1] or ".png"
        avatar_fn = f"avatar_{ts}{ext}"
        avatar_fp = str(avatar_dir / avatar_fn)
        shutil.copy2(final_abs, avatar_fp)

        import asyncio as _aio
        try:
            loop = _aio.new_event_loop()
            meta = _json.dumps({"group": group})
            loop.run_until_complete(record_user_file(user_id, "avatar", avatar_fn, avatar_fp, avatar_fn, meta))
            loop.close()
        except Exception as db_err:
            logging.error(f"Failed to record avatar to DB: {db_err}")

        logging.info(f"Avatar prepared and registered: {group}/{avatar_fn}")
        generation_status[task_id].update({
            "status": "completed", "progress": 1.0,
            "message": "Avatar prepared!", "output_path": final_output,
            "avatar_group": group, "avatar_filename": avatar_fn,
        })
    except Exception as e:
        logging.error(f"Avatar prepare failed: {e}")
        import traceback
        traceback.print_exc()
        generation_status[task_id].update({
            "status": "failed", "progress": 0,
            "message": str(e), "output_path": None,
        })
    finally:
        gpu1_lock.release()
        gpu0_lock.release()


@app.get("/api/fashion-styles")
async def get_fashion_styles(user=Depends(get_current_user)):
    """Load fashion styles from CSV."""
    import csv
    csv_path = Path("settings/fashion_hair/s1.csv")
    if not csv_path.exists():
        return {"styles": [], "categories": []}
    styles = []
    categories = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            styles.append({
                "id": int(row.get("ID", 0)),
                "category": row.get("Style_Keyword", ""),
                "prompt": row.get("English_Prompt", ""),
            })
            categories.add(row.get("Style_Keyword", ""))
    return {"styles": styles, "categories": sorted(categories)}


@app.post("/api/generate")
async def generate_video(request: GenerateRequest, user=Depends(get_current_user)):
    """Start S2V video generation task."""
    task_id = str(uuid.uuid4())

    generation_status[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Task queued",
        "output_path": None,
    }

    params = request.dict()
    params["_user_id"] = user["id"]
    executor.submit(generate_video_task, task_id, params)

    return {"task_id": task_id}


@app.post("/api/generate-i2v")
async def generate_i2v(request: I2VGenerateRequest, user=Depends(get_current_user)):
    """Start I2V video generation task."""
    if not os.path.exists(I2V_GGUF_HIGH) or not os.path.exists(I2V_GGUF_LOW):
        raise HTTPException(503, "I2V GGUF models not available. Download Q4_K_M GGUF files first.")

    task_id = str(uuid.uuid4())

    generation_status[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "I2V task queued",
        "output_path": None,
    }

    params = request.dict()
    params["_user_id"] = user["id"]
    executor.submit(generate_i2v_task, task_id, params)

    return {"task_id": task_id}


@app.get("/api/status/{task_id}")
async def get_status(task_id: str, user=Depends(get_current_user)):
    """Get generation task status."""
    if task_id not in generation_status:
        raise HTTPException(404, "Task not found")

    return generation_status[task_id]


@app.post("/api/cancel/{task_id}")
async def cancel_task(task_id: str, user=Depends(get_current_user)):
    """Cancel a running generation task."""
    if task_id not in generation_status:
        raise HTTPException(404, "Task not found")
    status = generation_status[task_id]
    if status["status"] in ("completed", "failed", "cancelled"):
        return {"ok": True, "message": "Task already finished"}
    # Interrupt ComfyUI and clear queue
    try:
        http_requests.post(f"{COMFYUI_URL}/interrupt", timeout=5)
        http_requests.post(f"{COMFYUI_URL}/queue", json={"clear": True}, timeout=5)
    except Exception:
        pass
    generation_status[task_id].update({
        "status": "cancelled",
        "message": "Cancelled by user",
    })
    return {"ok": True, "message": "Task cancelled"}


@app.post("/api/extract-audio")
async def extract_audio(video_path: str = Form(...), user=Depends(get_current_user)):
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
async def separate_vocals(audio_path: str = Form(...), user=Depends(get_current_user)):
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


class TrimVideoRequest(BaseModel):
    video_path: str
    start: float
    end: float

@app.post("/api/trim-video")
def trim_video(request: TrimVideoRequest, user=Depends(get_current_user)):
    """Trim video to specified start/end times using ffmpeg."""
    if not os.path.exists(request.video_path):
        raise HTTPException(404, f"Video not found: {request.video_path}")
    if request.start < 0 or request.end <= request.start:
        raise HTTPException(400, "Invalid start/end times")

    trimmed_path = str(UPLOAD_DIR / f"{uuid.uuid4()}.mp4")
    try:
        result = subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(request.start),
            "-to", str(request.end),
            "-i", request.video_path,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            trimmed_path,
        ], capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise Exception(f"ffmpeg failed: {result.stderr[:500]}")

        # Get duration of trimmed video
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", trimmed_path,
        ], capture_output=True, text=True, timeout=30)
        duration = float(probe.stdout.strip()) if probe.returncode == 0 else (request.end - request.start)

        # Delete original file after successful trim
        original = os.path.abspath(request.video_path)
        trimmed = os.path.abspath(trimmed_path)
        if original != trimmed and os.path.exists(original):
            os.remove(original)

        filename = os.path.basename(trimmed_path)
        return {
            "path": trimmed_path,
            "url": f"/uploads/{filename}",
            "duration": round(duration, 2),
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Trim operation timed out")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/uploads/images")
async def list_uploaded_images(user=Depends(get_current_user)):
    """List uploaded images owned by current user."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM user_files WHERE user_id = ? AND file_type = 'upload_image' ORDER BY created_at DESC",
            (user["id"],))
        rows = [dict(r) for r in await cursor.fetchall()]
    images = []
    for r in rows:
        fp = Path(r["file_path"])
        if fp.exists():
            meta = _json.loads(r["metadata"]) if r["metadata"] else {}
            images.append({
                "filename": r["filename"], "url": f"/uploads/{r['filename']}",
                "path": r["file_path"], "width": meta.get("width", 0), "height": meta.get("height", 0),
                "size": fp.stat().st_size, "created_at": r["created_at"],
            })
    return {"images": images, "total": len(images)}


@app.get("/api/uploads/audio")
async def list_uploaded_audio(user=Depends(get_current_user)):
    """List all uploaded audio files."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM user_files WHERE user_id = ? AND file_type = 'upload_audio' ORDER BY created_at DESC",
            (user["id"],))
        rows = [dict(r) for r in await cursor.fetchall()]
    audio_files = []
    for r in rows:
        fp = Path(r["file_path"])
        if fp.exists():
            audio_files.append({
                "filename": r["filename"], "url": f"/uploads/{r['filename']}",
                "path": r["file_path"], "size": fp.stat().st_size, "created_at": r["created_at"],
            })
    return {"audio": audio_files, "total": len(audio_files)}


@app.get("/api/videos")
async def list_videos(user=Depends(get_current_user)):
    """List generated videos owned by current user."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM user_files WHERE user_id = ? AND file_type = 'output' AND filename LIKE '%.mp4' ORDER BY created_at DESC",
            (user["id"],))
        rows = [dict(r) for r in await cursor.fetchall()]
    videos = []
    for r in rows:
        fp = Path(r["file_path"])
        if fp.exists():
            videos.append({
                "filename": r["filename"], "url": f"/outputs/{r['filename']}",
                "size": fp.stat().st_size, "created_at": r["created_at"],
            })
    return {"videos": videos, "total": len(videos)}


@app.delete("/api/videos/{filename}")
async def delete_video(filename: str, user=Depends(get_current_user)):
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


@app.get("/api/t2i-status")
async def t2i_status(user=Depends(get_current_user)):
    """Check if T2I (FLUX) model is available."""
    return {
        "available": True,
        "model": "FLUX.2-klein-9B",
        "upscale_available": os.path.exists(REALESRGAN_MODEL_PATH),
        "message": "FLUX.2-klein-9B ready. 4-step generation. First use requires model download.",
    }


@app.post("/api/generate-flux")
async def generate_flux(request: FluxGenerateRequest, user=Depends(get_current_user)):
    """Start FLUX image generation task."""
    task_id = str(uuid.uuid4())

    generation_status[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "FLUX task queued",
        "output_path": None,
    }

    params = request.dict()
    params["_user_id"] = user["id"]
    executor.submit(generate_flux_task, task_id, params)

    return {"task_id": task_id}


@app.delete("/api/outputs/{filename}")
async def delete_output(filename: str, user=Depends(get_current_user)):
    """Delete a generated output (image or video)."""
    filepath = OUTPUT_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, "File not found")
    if filepath.resolve().parent != OUTPUT_DIR.resolve():
        raise HTTPException(403, "Access denied")
    filepath.unlink()
    return {"message": f"Deleted {filename}"}


class YouTubeUploadRequest(BaseModel):
    filename: str
    title: str = ""
    description: str = ""
    hashtags: str = ""


@app.post("/api/upload-youtube")
def manual_upload_youtube(request: YouTubeUploadRequest, user=Depends(get_current_user)):
    """Manually upload a gallery video to YouTube."""
    filepath = OUTPUT_DIR / request.filename
    if not filepath.exists():
        raise HTTPException(404, "File not found")
    if filepath.resolve().parent != OUTPUT_DIR.resolve():
        raise HTTPException(403, "Access denied")
    ext = filepath.suffix.lower()
    if ext not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise HTTPException(400, "Only video files can be uploaded to YouTube")

    title = request.title.strip()
    if not title:
        title = f"AI Video - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"

    description = request.description.strip()
    tags = None
    if request.hashtags.strip():
        tags = [t.strip().lstrip("#") for t in request.hashtags.replace(",", " ").split() if t.strip()]
    if tags:
        hashtag_line = " ".join(f"#{t}" for t in tags)
        description = f"{description}\n\n{hashtag_line}".strip() if description else hashtag_line

    try:
        yt_url = upload_to_youtube(str(filepath), title, description, tags)
        if yt_url:
            try:
                notify_slack(f"YouTube upload: {yt_url}\nFile: {request.filename}")
            except Exception:
                pass
            return {"status": "ok", "youtube_url": yt_url}
        else:
            raise HTTPException(500, "YouTube upload returned no URL (check credentials)")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Manual YouTube upload failed: {e}")
        raise HTTPException(500, f"YouTube upload failed: {str(e)}")


@app.post("/api/extract-frame")
def extract_first_frame(video_path: str = Form(...), user=Depends(get_current_user)):
    """Extract first frame from a video file and save as PNG."""
    import cv2

    if not os.path.exists(video_path):
        raise HTTPException(404, "Video file not found")

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(500, "Failed to read video frame")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_filename = f"{timestamp}_frame.png"
    frame_path = UPLOAD_DIR / frame_filename

    cv2.imwrite(str(frame_path), frame)

    height, width = frame.shape[:2]

    return {
        "path": str(frame_path),
        "url": f"/uploads/{frame_filename}",
        "width": width,
        "height": height,
    }


@app.get("/api/outputs")
async def list_outputs(user=Depends(get_current_user)):
    """List generated outputs owned by current user."""
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM user_files WHERE user_id = ? AND file_type = 'output' ORDER BY created_at DESC",
            (user["id"],))
        rows = [dict(r) for r in await cursor.fetchall()]
    outputs = []
    for r in rows:
        fp = Path(r["file_path"])
        if fp.exists():
            ext = fp.suffix.lower()
            item = {
                "filename": r["filename"], "url": f"/outputs/{r['filename']}",
                "path": r["file_path"], "type": "image" if ext in IMAGE_EXTS else "video",
                "size": fp.stat().st_size, "created_at": r["created_at"],
            }
            if ext in IMAGE_EXTS:
                try:
                    with Image.open(fp) as img:
                        item["width"], item["height"] = img.size
                except Exception:
                    item["width"], item["height"] = 0, 0
            outputs.append(item)
    return {"outputs": outputs, "total": len(outputs)}


# ============================================================================
# ComfyUI Workflow Integration
# ============================================================================
comfyui_process = None


class WorkflowGenerateRequest(BaseModel):
    workflow_id: str
    inputs: dict = {}  # {key: value} matching registry input definitions
    yt_title: str = ""
    yt_description: str = ""
    yt_hashtags: str = ""


class YouTubeDownloadRequest(BaseModel):
    url: str


def ensure_comfyui_running():
    """Ensure ComfyUI server is running, start if needed."""
    global comfyui_process
    try:
        resp = http_requests.get(f"{COMFYUI_URL}/system_stats", timeout=3)
        if resp.status_code == 200:
            return
    except Exception:
        pass

    if not COMFYUI_DIR.exists():
        raise Exception(f"ComfyUI not installed at {COMFYUI_DIR}")

    logging.info("Starting ComfyUI server...")
    comfyui_log = open("/tmp/comfyui.log", "w")
    comfyui_process = subprocess.Popen(
        [sys.executable, str(COMFYUI_DIR / "main.py"),
         "--listen", "127.0.0.1", "--port", "8188",
         "--disable-auto-launch", "--preview-method", "none"],
        cwd=str(COMFYUI_DIR),
        stdout=comfyui_log, stderr=comfyui_log,
    )
    for _ in range(180):
        try:
            resp = http_requests.get(f"{COMFYUI_URL}/system_stats", timeout=2)
            if resp.status_code == 200:
                logging.info("ComfyUI server is ready!")
                return
        except Exception:
            pass
        _time.sleep(1)
    raise Exception("ComfyUI failed to start within 3 minutes")


def crop_face_head(image_path: str, padding_ratio: float = 1.8) -> str:
    """Crop image to head/face area to avoid clothing leaking into face swap.

    Uses mediapipe FaceDetector (tasks API) for accurate detection, then crops
    with generous padding to include hair and neck but exclude torso/clothing.
    Falls back to upper-half crop if no face is detected.
    Returns path to the cropped temporary image.
    """
    import mediapipe as mp
    import cv2
    import tempfile

    img = cv2.imread(image_path)
    if img is None:
        logging.warning(f"crop_face_head: could not read image {image_path}, returning as-is")
        return image_path

    h, w = img.shape[:2]

    # Use mediapipe tasks API for face detection
    model_path = str(Path(__file__).parent / "models" / "blaze_face_short_range.tflite")
    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        min_detection_confidence=0.5,
    )
    mp_image = mp.Image.create_from_file(image_path)

    with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
        result = detector.detect(mp_image)

    if result.detections:
        det = result.detections[0]
        bb = det.bounding_box
        fx, fy, fw, fh = bb.origin_x, bb.origin_y, bb.width, bb.height

        # Expand with padding_ratio (1.8x = generous head+hair+neck)
        cx, cy = fx + fw // 2, fy + fh // 2
        half_size = int(max(fw, fh) * padding_ratio / 2)

        x1 = max(0, cx - half_size)
        y1 = max(0, cy - int(half_size * 0.7))  # less padding above (forehead)
        x2 = min(w, cx + half_size)
        y2 = min(h, cy + int(half_size * 1.3))   # more padding below (neck)
        logging.info(f"crop_face_head: face detected at ({fx},{fy},{fw},{fh}) score={det.categories[0].score:.2f}, crop [{x1}:{x2}, {y1}:{y2}] from {w}x{h}")
    else:
        # Fallback: upper 50% of image (likely head area for full-body shots)
        x1, y1, x2, y2 = 0, 0, w, h // 2
        logging.warning(f"crop_face_head: no face detected, using upper half")

    cropped = img[y1:y2, x1:x2]
    suffix = os.path.splitext(image_path)[1] or ".png"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=str(OUTPUT_DIR))
    cv2.imwrite(tmp.name, cropped)
    tmp.close()
    logging.info(f"crop_face_head: saved cropped face to {tmp.name} ({x2-x1}x{y2-y1})")
    return tmp.name


def composite_face_onto_body(swapped_path: str, original_body_path: str, feather: int = 30) -> None:
    """Composite the face/head from swapped result onto the original body image.

    Detects face in the swapped result, creates a soft elliptical mask around
    the head area, and blends only that region onto the original body image
    (resized to match). This preserves original clothing/background exactly.
    Overwrites swapped_path in-place.
    """
    import mediapipe as mp
    import cv2
    import numpy as np

    swapped = cv2.imread(swapped_path)
    original = cv2.imread(original_body_path)
    if swapped is None or original is None:
        logging.warning("composite_face_onto_body: could not read images, skipping")
        return

    sh, sw = swapped.shape[:2]
    # Resize original to match swapped dimensions
    original = cv2.resize(original, (sw, sh), interpolation=cv2.INTER_LANCZOS4)

    # Detect face in the swapped result
    model_path = str(Path(__file__).parent / "models" / "blaze_face_short_range.tflite")
    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        min_detection_confidence=0.4,
    )
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB))

    with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
        result = detector.detect(mp_image)

    if not result.detections:
        logging.warning("composite_face_onto_body: no face in swapped result, skipping composite")
        return

    det = result.detections[0]
    bb = det.bounding_box
    fx, fy, fw, fh = bb.origin_x, bb.origin_y, bb.width, bb.height

    # Create elliptical mask centered on face with generous head padding
    cx, cy = fx + fw // 2, fy + fh // 2
    # Ellipse radii: wider horizontally (ears/hair), taller vertically (hair+chin+neck)
    rx = int(fw * 1.4)
    ry = int(fh * 1.8)

    mask = np.zeros((sh, sw), dtype=np.float32)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)

    # Feather the mask edges with Gaussian blur for smooth blending
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), feather)

    # Blend: face from swapped, body/background from original
    mask_3ch = mask[:, :, np.newaxis]
    blended = (swapped.astype(np.float32) * mask_3ch +
               original.astype(np.float32) * (1.0 - mask_3ch))
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    cv2.imwrite(swapped_path, blended)
    logging.info(f"composite_face_onto_body: face at ({fx},{fy},{fw},{fh}), ellipse ({rx},{ry}), feather={feather}")


def upload_to_comfyui(local_path: str, subfolder: str = "") -> str:
    """Upload a file to ComfyUI's input directory."""
    filename = os.path.basename(local_path)
    with open(local_path, 'rb') as f:
        resp = http_requests.post(
            f"{COMFYUI_URL}/upload/image",
            files={"image": (filename, f, "application/octet-stream")},
            data={"overwrite": "true", "subfolder": subfolder},
        )
    if resp.status_code != 200:
        raise Exception(f"ComfyUI upload failed: {resp.text}")
    return resp.json().get("name", filename)


def _set_nested(workflow: dict, node_id: str, field_path: str, value):
    """Set a value in a workflow node using dot-separated path like 'inputs.image'."""
    node = workflow.get(str(node_id))
    if not node:
        return
    parts = field_path.split(".")
    obj = node
    for p in parts[:-1]:
        obj = obj.setdefault(p, {})
    obj[parts[-1]] = value


def resolve_set_get_nodes(workflow: dict) -> dict:
    """
    Resolve SetNode/GetNode virtual nodes into direct connections.

    SetNode/GetNode are frontend-only JavaScript nodes in KJNodes with NO Python
    backend. They cannot be executed via ComfyUI's API. This function rewires all
    connections to bypass them and removes non-executable nodes.
    """
    # Class type display-name → registered-name fixes
    CLASS_TYPE_FIXES = {"Int": "easy int"}

    name_to_source = {}
    setnode_ids = set()
    getnode_ids = set()
    remove_ids = set()

    # Phase 1: Collect SetNode/GetNode/Note nodes
    for node_id, node in workflow.items():
        ct = node.get("class_type", "")
        if ct == "SetNode":
            setnode_ids.add(node_id)
            remove_ids.add(node_id)
            name = node["inputs"].get("value", "")
            for key, val in node["inputs"].items():
                if key != "value" and isinstance(val, list) and len(val) == 2:
                    name_to_source[name] = val
                    break
        elif ct == "GetNode":
            getnode_ids.add(node_id)
            remove_ids.add(node_id)
        elif ct in ("Note", "Fast Groups Bypasser (rgthree)"):
            remove_ids.add(node_id)

    if not setnode_ids and not getnode_ids:
        return workflow  # Nothing to resolve

    # Phase 2: Map GetNode → source, SetNode → source
    getnode_to_source = {}
    for nid in getnode_ids:
        name = workflow[nid]["inputs"].get("value", "")
        if name in name_to_source:
            getnode_to_source[nid] = name_to_source[name]

    setnode_to_source = {}
    for nid in setnode_ids:
        name = workflow[nid]["inputs"].get("value", "")
        if name in name_to_source:
            setnode_to_source[nid] = name_to_source[name]

    # Phase 3: Rewire all references
    for node_id, node in workflow.items():
        if node_id in remove_ids:
            continue
        for key, val in node.get("inputs", {}).items():
            if isinstance(val, list) and len(val) == 2:
                ref_id = str(val[0])
                if ref_id in getnode_to_source:
                    node["inputs"][key] = getnode_to_source[ref_id]
                elif ref_id in setnode_to_source:
                    node["inputs"][key] = setnode_to_source[ref_id]

    # Phase 4: Fix class_type mismatches
    for node_id, node in workflow.items():
        if node_id in remove_ids:
            continue
        ct = node.get("class_type", "")
        if ct in CLASS_TYPE_FIXES:
            node["class_type"] = CLASS_TYPE_FIXES[ct]

    # Phase 5: Remove virtual nodes
    for node_id in remove_ids:
        del workflow[node_id]

    logging.info(f"Resolved {len(setnode_ids)} SetNodes, {len(getnode_ids)} GetNodes, "
                 f"{len(remove_ids)} nodes removed")
    return workflow


def merge_audio_to_video(video_path: str, audio_source_path: str) -> str:
    """
    Extract audio from audio_source_path and merge it into video_path using ffmpeg.
    Returns path to the merged video, or original video_path if merge fails.
    """
    import subprocess

    # Only merge audio into video files
    VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    if os.path.splitext(video_path)[1].lower() not in VIDEO_EXTS:
        logging.warning(f"Skipping audio merge: not a video file ({video_path})")
        return video_path

    output_path = video_path.replace(".mp4", "_audio.mp4")
    # Safety: ensure output_path differs from input
    if output_path == video_path:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_audio{ext}"

    try:
        result = subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_source_path,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_path,
        ], capture_output=True, timeout=120)

        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logging.info(f"Audio merged successfully: {output_path}")
            # Replace original with merged version
            os.replace(output_path, video_path)
            return video_path
        else:
            logging.warning(f"ffmpeg audio merge failed: {result.stderr.decode()[:500]}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return video_path
    except Exception as e:
        logging.warning(f"Audio merge error: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return video_path


def upload_to_youtube(video_path: str, title: str, description: str = "",
                      tags: list = None) -> Optional[str]:
    """Upload video to YouTube as a Short. Returns video URL or None."""
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build as yt_build
    from googleapiclient.http import MediaFileUpload

    SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

    creds = None
    if YOUTUBE_TOKEN.exists():
        creds = Credentials.from_authorized_user_file(str(YOUTUBE_TOKEN), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not YOUTUBE_CLIENT_SECRET.exists():
                logging.warning("YouTube client_secret.json not found, skipping upload")
                return None
            flow = InstalledAppFlow.from_client_secrets_file(str(YOUTUBE_CLIENT_SECRET), SCOPES)
            creds = flow.run_local_server(port=0)
        with open(YOUTUBE_TOKEN, 'w') as f:
            f.write(creds.to_json())

    youtube = yt_build("youtube", "v3", credentials=creds)

    if tags is None:
        tags = ["Shorts", "AI", "dance", "korean"]
    if "Shorts" not in tags:
        tags.insert(0, "Shorts")

    body = {
        "snippet": {
            "title": title[:100],
            "description": description,
            "tags": tags,
            "categoryId": "22",
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(video_path, mimetype="video/mp4", resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = request.execute()

    video_id = response.get("id")
    video_url = f"https://youtube.com/shorts/{video_id}" if video_id else None
    logging.info(f"YouTube upload complete: {video_url}")
    return video_url


def notify_slack(message: str):
    """Send Slack notification via bot DM."""
    from slack_sdk import WebClient

    if not SLACK_BOT_TOKEN:
        logging.warning("SLACK_BOT_TOKEN not set, skipping notification")
        return
    if not SLACK_NOTIFY_EMAIL:
        logging.warning("SLACK_NOTIFY_EMAIL not set, skipping notification")
        return

    client = WebClient(token=SLACK_BOT_TOKEN)
    try:
        user = client.users_lookupByEmail(email=SLACK_NOTIFY_EMAIL)
        user_id = user["user"]["id"]
        dm = client.conversations_open(users=[user_id])
        channel_id = dm["channel"]["id"]
        client.chat_postMessage(channel=channel_id, text=message)
        logging.info(f"Slack notification sent to {SLACK_NOTIFY_EMAIL}")
    except Exception as e:
        logging.error(f"Slack notification failed: {e}")


def prepare_comfyui_workflow(workflow_id: str, user_inputs: dict) -> dict:
    """Load API-format workflow and inject user parameters using the registry."""
    import json as json_mod
    wf_config = WORKFLOW_REGISTRY.get(workflow_id)
    if not wf_config:
        raise Exception(f"Unknown workflow: {workflow_id}")

    api_path = WORKFLOW_DIR / wf_config["api_json"]
    with open(api_path) as f:
        workflow = json_mod.load(f)

    # Resolve SetNode/GetNode virtual nodes (frontend-only, no Python backend)
    workflow = resolve_set_get_nodes(workflow)

    for input_def in wf_config["inputs"]:
        key = input_def["key"]
        value = user_inputs.get(key, input_def.get("default"))
        if value is None:
            continue

        # Skip special marker node_ids (handled in post-processing below)
        node_id = input_def.get("node_id", "")
        if isinstance(node_id, str) and node_id.startswith("__"):
            continue

        # Skip inputs without node_id (metadata-only, handled in post-processing)
        if not node_id and "node_ids" not in input_def:
            continue

        if input_def["type"] == "select_buttons":
            option = next((o for o in input_def["options"] if o["value"] == value), None)
            if option and "params" in option:
                for param_key, param_val in option["params"].items():
                    _set_nested(workflow, input_def["node_ids"][param_key], input_def["fields"][param_key], param_val)
        elif "node_ids" in input_def:
            for nid in input_def["node_ids"]:
                _set_nested(workflow, nid, input_def["field"], value)
        else:
            _set_nested(workflow, input_def["node_id"], input_def["field"], value)

    # --- Change Character: Custom Background ---
    if workflow_id == "change_character":
        bg_image = user_inputs.get("bg_image")
        bg_prompt = user_inputs.get("bg_prompt", "").strip()

        # Custom background image → route through masking pipeline
        # Original flow: input_video(81) → DrawMaskOnImage(15) → bg_images(218)
        # Custom flow:   LoadImage(300) → Resize(301) → Repeat(302) → DrawMaskOnImage(15) → bg_images(218)
        # This ensures the character area is properly masked (blacked out) in the custom background
        if bg_image:
            workflow["300"] = {
                "class_type": "LoadImage",
                "inputs": {"image": bg_image, "upload": "image"},
            }
            workflow["301"] = {
                "class_type": "ImageResizeKJv2",
                "inputs": {
                    "width": ["123", 0],
                    "height": ["124", 0],
                    "upscale_method": "lanczos",
                    "keep_proportion": "pad",
                    "pad_color": "0, 0, 0",
                    "crop_position": "center",
                    "divisible_by": 16,
                    "device": "cpu",
                    "image": ["300", 0],
                },
            }
            workflow["302"] = {
                "class_type": "RepeatImageBatch",
                "inputs": {
                    "image": ["301", 0],
                    "amount": ["96", 1],
                },
            }
            # Replace DrawMaskOnImage's image input with repeated custom bg
            # Node 15 still applies the character mask, producing bg with character area blacked out
            if "15" in workflow:
                workflow["15"]["inputs"]["image"] = ["302", 0]
                logging.info(f"Custom background image applied: {bg_image}")

        # Background prompt → prepend to positive_prompt
        if bg_prompt and "209" in workflow:
            current = workflow["209"]["inputs"].get("positive_prompt", "")
            workflow["209"]["inputs"]["positive_prompt"] = f"{bg_prompt}. {current}"
            logging.info(f"Background prompt prepended: {bg_prompt}")

    # --- Fashion Change: seed randomization + optional clothing ref ---
    if workflow_id == "fashion_change":
        # Random seed if -1
        seed_val = workflow.get("113", {}).get("inputs", {}).get("value", -1)
        if seed_val == -1:
            workflow["113"]["inputs"]["value"] = random.randint(0, 2**53)

        # If no clothing reference image provided, remove ref image nodes
        # and simplify the pipeline to text-only inpainting
        clothing_ref = user_inputs.get("clothing_ref", "")
        if not clothing_ref:
            # Remove clothing reference nodes
            for nid in ["105", "106:85", "106:84:78", "106:84:77", "106:84:76"]:
                workflow.pop(nid, None)
            # CFGGuider gets conditioning from cropped-image ReferenceLatent nodes
            if "106:63" in workflow:
                workflow["106:63"]["inputs"]["positive"] = ["106:79:77", 0]
                workflow["106:63"]["inputs"]["negative"] = ["106:79:76", 0]
            logging.info("Fashion change: text-only mode (no clothing reference image)")
        else:
            # When clothing reference is provided, inject Try-On LoRA if available
            tryon_lora = Path(COMFYUI_DIR) / "models" / "loras" / "KleinBase9B_TryOn.safetensors"
            if tryon_lora.exists():
                # Insert LoraLoaderModelOnly between UNETLoader (106:70) and CFGGuider (106:63)
                workflow["tryon_lora"] = {
                    "inputs": {
                        "model": ["106:70", 0],
                        "lora_name": "KleinBase9B_TryOn.safetensors",
                        "strength_model": 1.5,
                    },
                    "class_type": "LoraLoaderModelOnly",
                    "_meta": {"title": "Try-On LoRA"},
                }
                # CFGGuider now uses LoRA-patched model instead of raw UNETLoader
                if "106:63" in workflow:
                    workflow["106:63"]["inputs"]["model"] = ["tryon_lora", 0]
                # Update prompt to use trigger word for Try-On LoRA
                if "106:74" in workflow:
                    original_prompt = workflow["106:74"]["inputs"].get("text", "")
                    workflow["106:74"]["inputs"]["text"] = f"attach the outfit in Image 2 to the woman in Image 1. {original_prompt}"
                logging.info(f"Fashion change: Try-On LoRA applied with clothing ref: {clothing_ref}")
            else:
                logging.info(f"Fashion change: reference mode (no Try-On LoRA found): {clothing_ref}")

    # --- Face Swap: ethnicity injection + seed randomization ---
    if workflow_id == "face_swap":
        ethnicity = user_inputs.get("ethnicity", "korean")
        if ethnicity and ethnicity != "auto":
            # Korean-specific: ulzzang style (Korean beauty standard)
            eth_prompts = {
                "korean": {
                    "positive": " The person in image 2 is Korean. Preserve Korean ulzzang facial features: soft jawline, natural double eyelids, warm ivory skin tone, small nose bridge, full lips. The result must look like a real Korean person.",
                    "negative": ", western features, caucasian features, european face",
                },
                "asian": {
                    "positive": " The person in image 2 is East Asian. Preserve East Asian facial features, skin tone, and appearance.",
                    "negative": ", western features, caucasian features, european face",
                },
                "western": {
                    "positive": " The person in image 2 is Western/Caucasian. Preserve Western facial features, skin tone, and appearance.",
                    "negative": ", asian features, east asian features",
                },
            }
            eth = eth_prompts.get(ethnicity, eth_prompts["korean"])

            if "50" in workflow:
                workflow["50"]["inputs"]["text"] += eth["positive"]
            if "51" in workflow:
                workflow["51"]["inputs"]["text"] += eth["negative"]

            logging.info(f"Face swap: ethnicity={ethnicity} injected into prompts")

        seed_val = workflow.get("92", {}).get("inputs", {}).get("value", -1)
        if seed_val == -1:
            workflow["92"]["inputs"]["value"] = random.randint(0, 2**53)

    # --- Scene Composite: seed randomization ---
    if workflow_id == "scene_composite":
        seed_val = workflow.get("90", {}).get("inputs", {}).get("seed", -1)
        if seed_val == -1:
            workflow["90"]["inputs"]["seed"] = random.randint(0, 2**53)

    return workflow


def monitor_comfyui_progress(task_id: str, prompt_id: str, client_id: str):
    """Monitor ComfyUI execution via WebSocket."""
    ws = ws_client.WebSocket()
    ws.settimeout(1800)  # 30 min timeout
    ws.connect(f"{COMFYUI_WS_URL}?clientId={client_id}")
    import json as json_mod

    completed_nodes = 0

    try:
        while True:
            msg = ws.recv()
            if isinstance(msg, str):
                data = json_mod.loads(msg)
                msg_type = data.get("type")
                msg_data = data.get("data", {})

                if msg_type == "progress":
                    value = msg_data.get("value", 0)
                    max_val = msg_data.get("max", 1)
                    node_pct = value / max_val if max_val > 0 else 0
                    overall = 0.15 + 0.75 * (completed_nodes / 78.0) + 0.75 * node_pct / 78.0
                    generation_status[task_id]["progress"] = round(min(overall, 0.90), 3)
                    generation_status[task_id]["message"] = (
                        f"Processing... node {completed_nodes}/78 (step {value}/{max_val})"
                    )
                elif msg_type == "executing":
                    node_id = msg_data.get("node")
                    if node_id is None and msg_data.get("prompt_id") == prompt_id:
                        break
                    completed_nodes += 1
                    overall = 0.15 + 0.75 * (completed_nodes / 78.0)
                    generation_status[task_id]["progress"] = round(min(overall, 0.90), 3)
                    generation_status[task_id]["message"] = f"Executing node {completed_nodes}/78..."
                elif msg_type == "execution_error":
                    error = msg_data.get("exception_message", "Unknown ComfyUI error")
                    raise Exception(f"ComfyUI error: {error}")
    finally:
        ws.close()


def retrieve_comfyui_output(prompt_id: str) -> str:
    """Retrieve the output video from ComfyUI history."""
    import json as json_mod
    resp = http_requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
    history = resp.json()

    outputs = history.get(prompt_id, {}).get("outputs", {})

    # Pass 1: Prioritize video outputs (VHS_VideoCombine 'gifs' key)
    for node_id, node_output in outputs.items():
        if "gifs" in node_output:
            for gif in node_output["gifs"]:
                filename = gif["filename"]
                subfolder = gif.get("subfolder", "")
                file_type = gif.get("type", "temp")

                view_resp = http_requests.get(
                    f"{COMFYUI_URL}/view",
                    params={"filename": filename, "subfolder": subfolder, "type": file_type},
                )
                if view_resp.status_code != 200:
                    continue

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"workflow_{timestamp}.mp4"
                output_path = OUTPUT_DIR / output_filename
                with open(output_path, 'wb') as f:
                    f.write(view_resp.content)

                return f"/outputs/{output_filename}"

    # Pass 2: Fall back to final output images (skip temp previews)
    for node_id, node_output in outputs.items():
        if "images" in node_output:
            for img in node_output["images"]:
                if img.get("type") == "temp":
                    continue  # skip preview/temp images
                filename = img["filename"]
                subfolder = img.get("subfolder", "")
                file_type = img.get("type", "output")

                view_resp = http_requests.get(
                    f"{COMFYUI_URL}/view",
                    params={"filename": filename, "subfolder": subfolder, "type": file_type},
                )
                if view_resp.status_code != 200:
                    continue

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                ext = os.path.splitext(filename)[1] or ".png"
                output_filename = f"workflow_{timestamp}{ext}"
                output_path = OUTPUT_DIR / output_filename
                with open(output_path, 'wb') as f:
                    f.write(view_resp.content)

                return f"/outputs/{output_filename}"

    raise Exception("No output found in ComfyUI history")


def download_youtube_video(url: str) -> dict:
    """Download a YouTube video using yt-dlp and return the file path.

    Prefers H.264 (avc1) for OpenCV/ComfyUI compatibility.
    Falls back to any codec + ffmpeg re-encode if H.264 unavailable.
    Limits resolution to 1080p (max 1920 on either dimension).
    """
    import yt_dlp

    filename = f"{uuid.uuid4()}.mp4"
    filepath = UPLOAD_DIR / filename

    # Prefer H.264 (avc1) for ComfyUI/OpenCV compatibility, limit to 1080p
    ydl_opts = {
        'format': (
            'bestvideo[height<=1920][width<=1920][vcodec^=avc1][ext=mp4]+bestaudio[ext=m4a]/'
            'bestvideo[height<=1920][width<=1920][ext=mp4]+bestaudio[ext=m4a]/'
            'bestvideo[height<=1920][width<=1920]+bestaudio/'
            'best[ext=mp4]/best'
        ),
        'merge_output_format': 'mp4',
        'outtmpl': str(filepath),
        'quiet': True,
        'no_warnings': True,
        'socket_timeout': 30,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get('title', 'Unknown')
        vcodec = info.get('vcodec', '')

    # yt-dlp may add extensions; find the actual file
    actual_path = filepath
    if not filepath.exists():
        for f in UPLOAD_DIR.glob(f"{filepath.stem}*"):
            actual_path = f
            break

    # Re-encode to H.264 if downloaded codec is not avc1 (OpenCV compatibility)
    if vcodec and not vcodec.startswith('avc1'):
        logging.info(f"Re-encoding {vcodec} → H.264 for OpenCV compatibility...")
        h264_path = UPLOAD_DIR / f"{filepath.stem}_h264.mp4"
        try:
            result = subprocess.run([
                'ffmpeg', '-y', '-i', str(actual_path),
                '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
                '-c:a', 'aac', '-b:a', '192k',
                str(h264_path),
            ], capture_output=True, timeout=120)
            if result.returncode == 0 and h264_path.exists() and h264_path.stat().st_size > 0:
                actual_path.unlink()
                h264_path.rename(actual_path)
                logging.info(f"Re-encode complete: {actual_path}")
            else:
                logging.warning(f"Re-encode failed: {result.stderr.decode()[:200]}")
                if h264_path.exists():
                    h264_path.unlink()
        except Exception as e:
            logging.warning(f"Re-encode error: {e}")
            if h264_path.exists():
                h264_path.unlink()

    return {
        "path": str(actual_path),
        "url": f"/uploads/{actual_path.name}",
        "filename": actual_path.name,
        "title": title,
    }


def workflow_generate_task(task_id: str, params: dict):
    """Background task for ComfyUI workflow execution."""
    import json as json_mod

    gpu0_lock.acquire()
    gpu1_lock.acquire()

    try:
        workflow_id = params["workflow_id"]
        user_inputs = dict(params.get("inputs", {}))
        wf_config = WORKFLOW_REGISTRY[workflow_id]

        # Step 1: Offload all WanAvatar models
        generation_status[task_id].update({
            "status": "processing", "progress": 0.02,
            "message": "Offloading models for ComfyUI...",
        })
        _offload_s2v()
        _offload_i2v()
        _offload_flux()
        gc.collect()
        torch.cuda.empty_cache()

        # Step 2: Ensure ComfyUI is running
        generation_status[task_id].update({"progress": 0.05, "message": "Starting ComfyUI..."})
        ensure_comfyui_running()

        # Step 3: Upload files to ComfyUI as needed
        # For face_swap: save original body path for post-composite, crop avatar face
        original_body_path = None
        cropped_tmp = None
        if workflow_id == "face_swap":
            # Resolve style_source to absolute path for later face composite
            style_src = user_inputs.get("style_source", "")
            if style_src:
                for d in [OUTPUT_DIR, UPLOAD_DIR, AVATARS_DIR]:
                    candidate = d / os.path.basename(style_src)
                    if candidate.exists():
                        original_body_path = str(candidate)
                        break
                if not original_body_path and os.path.isabs(style_src) and os.path.exists(style_src):
                    original_body_path = style_src
        if workflow_id == "face_swap" and user_inputs.get("avatar_face"):
            generation_status[task_id].update({"progress": 0.07, "message": "Cropping face from avatar..."})
            cropped_tmp = crop_face_head(user_inputs["avatar_face"])
            user_inputs["avatar_face"] = cropped_tmp

        for input_def in wf_config["inputs"]:
            key = input_def["key"]
            if input_def.get("upload_to_comfyui") and key in user_inputs and user_inputs[key]:
                generation_status[task_id].update({"progress": 0.08, "message": f"Uploading {key}..."})
                user_inputs[key] = upload_to_comfyui(user_inputs[key])

        # Clean up cropped temp file after upload
        if cropped_tmp and os.path.exists(cropped_tmp):
            try:
                os.remove(cropped_tmp)
            except OSError:
                pass

        # Step 4: Prepare workflow
        generation_status[task_id].update({"progress": 0.12, "message": "Preparing workflow..."})
        logging.info(f"[{workflow_id}] user_inputs before prepare: { {k:v for k,v in user_inputs.items() if isinstance(v, str) and len(v) < 200} }")
        workflow = prepare_comfyui_workflow(workflow_id, user_inputs)
        if workflow_id == "face_swap":
            logging.info(f"[face_swap] node10(Picture1/body)={workflow.get('10',{}).get('inputs',{}).get('image','?')}  node11(Picture2/face)={workflow.get('11',{}).get('inputs',{}).get('image','?')}")

        # Step 5: Submit workflow
        generation_status[task_id].update({"progress": 0.15, "message": "Submitting workflow..."})
        client_id = str(uuid.uuid4())
        resp = http_requests.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow, "client_id": client_id},
        )
        if resp.status_code != 200:
            raise Exception(f"ComfyUI rejected workflow: {resp.text}")
        prompt_id = resp.json()["prompt_id"]

        # Step 6: Monitor progress
        generation_status[task_id].update({"message": "Executing workflow..."})
        try:
            monitor_comfyui_progress(task_id, prompt_id, client_id)
        except Exception as monitor_err:
            # WebSocket timeout — poll ComfyUI history as fallback
            logging.warning(f"Monitor timed out, polling ComfyUI for completion: {monitor_err}")
            generation_status[task_id].update({"progress": 0.90, "message": "Monitor lost connection, checking result..."})
            import time as _time
            for _attempt in range(60):  # poll up to 10 min (60 × 10s)
                _time.sleep(10)
                try:
                    _hist = http_requests.get(f"{COMFYUI_URL}/history/{prompt_id}").json()
                    _outs = _hist.get(prompt_id, {}).get("outputs", {})
                    if _outs:
                        logging.info(f"ComfyUI completed (detected via polling after {_attempt*10}s)")
                        break
                except Exception:
                    pass
            else:
                raise Exception(f"ComfyUI execution timed out after monitor failure: {monitor_err}")

        # Step 7: Retrieve output
        generation_status[task_id].update({"progress": 0.92, "message": "Retrieving output..."})
        output_path = retrieve_comfyui_output(prompt_id)

        # Step 7.5: Post-processing — face composite for face_swap
        # Blend only the swapped face onto the original body to preserve clothing/background
        if workflow_id == "face_swap" and original_body_path:
            abs_output_tmp = str(OUTPUT_DIR / os.path.basename(output_path))
            if os.path.exists(abs_output_tmp):
                generation_status[task_id].update({"progress": 0.93, "message": "Compositing face onto original body..."})
                try:
                    composite_face_onto_body(abs_output_tmp, original_body_path)
                except Exception as comp_err:
                    logging.warning(f"Face composite failed (using raw result): {comp_err}")

        # Step 7.6: Post-processing — scene composite (Z-Image + ControlNet)
        # If scene_background was provided, run Z-Image scene composite on the result
        scene_bg = user_inputs.get("scene_background")
        scene_prompt = user_inputs.get("scene_prompt", "A person standing naturally in the scene, photorealistic, natural lighting, correct perspective and depth, 4K")
        if scene_bg and workflow_id == "fashion_change":
            abs_output_for_scene = str(OUTPUT_DIR / os.path.basename(output_path))
            if os.path.exists(abs_output_for_scene):
                generation_status[task_id].update({"progress": 0.94, "message": "Scene composite: uploading images..."})
                try:
                    import json as json_mod2
                    # Upload character result to ComfyUI; scene_bg is already uploaded
                    char_comfyui = upload_to_comfyui(abs_output_for_scene)
                    bg_comfyui = scene_bg  # already uploaded via upload_to_comfyui in Step 3

                    # Load scene composite workflow
                    scene_wf_path = WORKFLOW_DIR / "z_image_scene_api.json"
                    with open(scene_wf_path) as f:
                        scene_wf = json_mod2.load(f)

                    scene_wf["10"]["inputs"]["image"] = char_comfyui
                    scene_wf["11"]["inputs"]["image"] = bg_comfyui
                    scene_wf["50"]["inputs"]["user_prompt"] = scene_prompt
                    scene_wf["90"]["inputs"]["seed"] = random.randint(0, 2**53)

                    generation_status[task_id].update({"progress": 0.95, "message": "Scene composite: generating..."})
                    client_id_sc = str(uuid.uuid4())
                    resp_sc = http_requests.post(
                        f"{COMFYUI_URL}/prompt",
                        json={"prompt": scene_wf, "client_id": client_id_sc},
                    )
                    if resp_sc.status_code == 200:
                        prompt_id_sc = resp_sc.json()["prompt_id"]
                        try:
                            monitor_comfyui_progress(task_id, prompt_id_sc, client_id_sc)
                        except Exception:
                            import time as _time_sc
                            for _ in range(60):
                                _time_sc.sleep(10)
                                try:
                                    h_sc = http_requests.get(f"{COMFYUI_URL}/history/{prompt_id_sc}").json()
                                    if h_sc.get(prompt_id_sc, {}).get("outputs", {}):
                                        break
                                except Exception:
                                    pass

                        scene_output = retrieve_comfyui_output(prompt_id_sc)
                        # Replace the original output with scene composite result
                        import shutil
                        shutil.copy2(str(OUTPUT_DIR / os.path.basename(scene_output)), abs_output_for_scene)
                        logging.info(f"Scene composite applied: {scene_output} -> {abs_output_for_scene}")
                    else:
                        logging.warning(f"Scene composite workflow rejected: {resp_sc.text}")
                except Exception as scene_err:
                    logging.warning(f"Scene composite failed (using original result): {scene_err}")

        # Step 8: Post-processing — merge audio (video outputs only)
        # Priority: custom_audio > reference video audio
        custom_audio = params.get("_custom_audio_path")
        ref_video_original = params.get("_ref_video_original")
        abs_output = str(OUTPUT_DIR / os.path.basename(output_path))
        is_video_output = os.path.splitext(output_path)[1].lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        if is_video_output and custom_audio and os.path.exists(custom_audio):
            generation_status[task_id].update({"progress": 0.95, "message": "Merging custom audio..."})
            merge_audio_to_video(abs_output, custom_audio)
            logging.info(f"Custom audio merged: {custom_audio}")
        elif is_video_output and ref_video_original and os.path.exists(ref_video_original):
            generation_status[task_id].update({"progress": 0.95, "message": "Merging audio from reference video..."})
            merge_audio_to_video(abs_output, ref_video_original)
            logging.info(f"Audio merged from reference video: {ref_video_original}")

        # Step 9: Post-processing — YouTube upload + Slack notification
        # Skip for image-only workflows (fashion_change)
        output_type = wf_config.get("output_type", "video")
        if output_type == "video" and os.path.exists(abs_output):
            generation_status[task_id].update({"progress": 0.97, "message": "Uploading to YouTube..."})

            yt_title = params.get("yt_title", "").strip()
            yt_description = params.get("yt_description", "").strip()
            yt_hashtags_raw = params.get("yt_hashtags", "").strip()

            # Parse hashtags into tags list
            yt_tags = None
            if yt_hashtags_raw:
                yt_tags = [t.strip().lstrip("#") for t in yt_hashtags_raw.replace(",", " ").split() if t.strip()]

            # Auto-generate title if empty: "{avatar_name} - YYYY-MM-DD HH:MM"
            if not yt_title:
                import datetime as dt_mod
                avatar_name = ""
                ref_image_path = user_inputs.get("ref_image", "")
                if ref_image_path and "settings/avatars/" in ref_image_path:
                    parts = ref_image_path.split("settings/avatars/")
                    if len(parts) > 1:
                        avatar_name = parts[1].split("/")[0]
                if not avatar_name:
                    avatar_name = "AI Video"
                now_str = dt_mod.datetime.now().strftime("%Y-%m-%d %H:%M")
                yt_title = f"{avatar_name} - {now_str}"

            # Append hashtags to description
            if yt_tags:
                hashtag_line = " ".join(f"#{t}" for t in yt_tags)
                yt_description = f"{yt_description}\n\n{hashtag_line}".strip() if yt_description else hashtag_line

            yt_url = None
            try:
                yt_url = upload_to_youtube(abs_output, yt_title, yt_description, yt_tags)
            except Exception as e:
                logging.error(f"YouTube upload failed: {e}")

            try:
                slack_msg = f"Video generation completed!\n"
                if yt_url:
                    slack_msg += f"YouTube: {yt_url}\n"
                slack_msg += f"File: {os.path.basename(abs_output)}"
                notify_slack(slack_msg)
            except Exception as e:
                logging.error(f"Slack notify failed: {e}")

        # Record output file to user_files DB
        _user_id = params.get("_user_id")
        if _user_id and output_path:
            out_fn = os.path.basename(output_path)
            out_fp = str(OUTPUT_DIR / out_fn)
            import asyncio as _aio
            try:
                loop = _aio.new_event_loop()
                loop.run_until_complete(record_user_file(_user_id, "output", out_fn, out_fp, out_fn))
                loop.close()
            except Exception as db_err:
                logging.error(f"Failed to record output to DB: {db_err}")

        generation_status[task_id].update({
            "status": "completed", "progress": 1.0,
            "message": "Workflow completed!", "output_path": output_path,
        })
    except Exception as e:
        logging.error(f"Workflow generation failed: {e}")
        import traceback
        traceback.print_exc()
        generation_status[task_id].update({
            "status": "failed", "progress": 0,
            "message": str(e), "output_path": None,
        })
    finally:
        gpu1_lock.release()
        gpu0_lock.release()


@app.post("/api/workflow/generate")
async def start_workflow_generation(request: WorkflowGenerateRequest, user=Depends(get_current_user)):
    """Start ComfyUI workflow generation task."""
    wf_config = WORKFLOW_REGISTRY.get(request.workflow_id)
    if not wf_config:
        raise HTTPException(400, f"Unknown workflow: {request.workflow_id}")

    api_path = WORKFLOW_DIR / wf_config["api_json"]
    if not api_path.exists():
        raise HTTPException(503, f"Workflow API JSON not found: {wf_config['api_json']}")

    task_id = str(uuid.uuid4())
    generation_status[task_id] = {
        "status": "pending", "progress": 0,
        "message": "Workflow task queued", "output_path": None,
    }

    params = request.dict()

    # Save original reference video path for audio extraction post-processing
    # (before upload_to_comfyui changes the filename)
    for input_def in wf_config["inputs"]:
        if input_def["type"] == "video" and input_def["key"] in (params.get("inputs") or {}):
            ref_path = params["inputs"][input_def["key"]]
            if ref_path:
                # Resolve to absolute path
                abs_ref = ref_path
                if not os.path.isabs(ref_path):
                    abs_ref = str(Path(ref_path).resolve())
                params["_ref_video_original"] = abs_ref
                break

    # Save custom audio path (for audio replacement in post-processing)
    custom_audio = (params.get("inputs") or {}).get("custom_audio")
    if custom_audio:
        abs_audio = custom_audio
        if not os.path.isabs(custom_audio):
            abs_audio = str(Path(custom_audio).resolve())
        params["_custom_audio_path"] = abs_audio

    params["_user_id"] = user["id"]
    executor.submit(workflow_generate_task, task_id, params)
    return {"task_id": task_id}


@app.get("/api/workflows")
async def list_workflows(user=Depends(get_current_user)):
    """Return available workflow definitions for the frontend."""
    result = []
    for wf in WORKFLOW_REGISTRY.values():
        api_path = WORKFLOW_DIR / wf["api_json"]
        if api_path.exists():
            result.append({
                "id": wf["id"],
                "display_name": wf["display_name"],
                "description": wf["description"],
                "inputs": wf["inputs"],
                "output_type": wf.get("output_type", "video"),
            })
    return {"workflows": result}


@app.post("/api/workflow/prepare-images")
async def prepare_workflow_images(request: dict, user=Depends(get_current_user)):
    """Copy selected gallery images to a temp folder for workflow use."""
    import shutil
    image_paths = request.get("image_paths", [])
    if not image_paths:
        raise HTTPException(400, "No image paths provided")
    folder_name = str(uuid.uuid4())
    folder_path = UPLOAD_DIR / "workflow_images" / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(image_paths):
        src_path = Path(src)
        if src_path.exists():
            ext = src_path.suffix
            dst = folder_path / f"{i:04d}{ext}"
            shutil.copy2(str(src_path), str(dst))
    return {"folder_path": str(folder_path), "count": len(image_paths)}


@app.get("/api/workflow/status")
async def workflow_comfyui_status(user=Depends(get_current_user)):
    """Check if ComfyUI is available."""
    available = False
    try:
        resp = http_requests.get(f"{COMFYUI_URL}/system_stats", timeout=3)
        available = resp.status_code == 200
    except Exception:
        pass
    available_workflows = [wf_id for wf_id, wf in WORKFLOW_REGISTRY.items()
                           if (WORKFLOW_DIR / wf["api_json"]).exists()]
    return {
        "comfyui_available": available,
        "comfyui_installed": COMFYUI_DIR.exists(),
        "available_workflows": available_workflows,
    }


@app.post("/api/download-youtube")
def download_youtube(request: YouTubeDownloadRequest, user=Depends(get_current_user)):
    """Download a YouTube video for use as reference.
    Note: plain def (not async) so FastAPI runs it in a threadpool,
    preventing the event loop from blocking during yt-dlp download.
    """
    try:
        result = download_youtube_video(request.url)
        return result
    except Exception as e:
        raise HTTPException(500, f"YouTube download failed: {str(e)}")


# ============================================================================
# Video Studio: TTS + Gemini AI Chat
# ============================================================================
tts_model = None


def get_tts_model():
    """Lazy-load Qwen3-TTS model on first use."""
    global tts_model
    if tts_model is None:
        from qwen_tts import Qwen3TTSModel
        tts_gpu = 1 if torch.cuda.device_count() >= 2 else 0
        logging.info(f"Loading Qwen3-TTS on cuda:{tts_gpu}...")
        tts_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            device_map=f"cuda:{tts_gpu}",
            dtype=torch.bfloat16,
        )
        logging.info("Qwen3-TTS loaded!")
    return tts_model


class TTSRequest(BaseModel):
    text: str
    language: str = "Korean"
    speaker: str = "Ryan"


@app.post("/api/studio/tts")
async def studio_tts(request: TTSRequest, user=Depends(get_current_user)):
    """Generate speech audio from text script using Qwen3-TTS."""
    import soundfile as sf
    try:
        model = get_tts_model()
        wavs, sr = model.generate_custom_voice(
            text=request.text,
            language=request.language,
            speaker=request.speaker,
        )
        filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
        audio_dir = UPLOAD_DIR / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = str(audio_dir / filename)
        sf.write(audio_path, wavs[0], sr)
        duration = len(wavs[0]) / sr
        return {
            "audio_path": audio_path,
            "audio_url": f"/uploads/audio/{filename}",
            "duration": round(duration, 2),
            "frame_count": int(duration * 20),
        }
    except Exception as e:
        logging.error(f"TTS generation failed: {e}")
        raise HTTPException(500, f"TTS failed: {str(e)}")


@app.get("/api/studio/tts-speakers")
async def get_tts_speakers(user=Depends(get_current_user)):
    """Return available TTS speakers for CustomVoice model."""
    return {
        "speakers": ["Ryan", "Claire", "Laura", "Aidan", "Matt", "Aria",
                      "Serena", "Leo", "Mei", "Luna"],
        "languages": ["Korean", "English", "Chinese", "Japanese",
                       "German", "French", "Russian", "Portuguese",
                       "Spanish", "Italian"],
    }


# --- Gemini AI Chat for Video Studio ---
STUDIO_TOOLS = [
    {
        "function_declarations": [
            {
                "name": "search_lora",
                "description": "Search LoRA adapters by style, character type, or motion. Returns matching LoRAs with trigger words and recommended weights.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language style description, e.g. 'sexy dance outfit' or 'cute pastel aesthetic'"},
                        "category": {"type": "string", "enum": ["img", "mov"], "description": "img for image LoRAs, mov for video/motion LoRAs"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "generate_idol_image",
                "description": "Generate an image using FLUX with specified LoRA and prompt. Use same seed for character consistency across frames.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "seed": {"type": "integer", "description": "Use same seed for consistent character across multiple images"},
                        "aspect_ratio": {"type": "string", "enum": ["portrait", "landscape", "square"]},
                        "lora_weights": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "weight": {"type": "number"},
                                },
                            },
                        },
                    },
                    "required": ["prompt"],
                },
            },
            {
                "name": "create_video",
                "description": "Create a video. Three modes: 'fflf' creates video from image sequence (dance/presentation); 'infinitalk' creates narration video from single image + audio (unlimited duration); 'change_character' replaces character in a reference video.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "enum": ["fflf", "infinitalk", "change_character"]},
                        "content_type": {"type": "string", "enum": ["dance", "narration", "presentation"]},
                        "image_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "For fflf: ordered image paths. For infinitalk/change_character: single image path.",
                        },
                        "segment_lengths": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Frame count per segment (fflf only, 17-81). Defaults: dance=33, presentation=49",
                        },
                        "master_prompt": {"type": "string"},
                        "seed": {"type": "integer"},
                        "looping": {"type": "boolean"},
                        "audio_path": {"type": "string", "description": "Audio file path (infinitalk mode, required)"},
                        "ref_video_url": {"type": "string", "description": "YouTube URL or video path (change_character mode)"},
                        "aspect_ratio": {"type": "string", "enum": ["portrait", "landscape", "square"]},
                    },
                    "required": ["mode", "image_paths"],
                },
            },
            {
                "name": "generate_tts",
                "description": "Generate speech audio from text using Qwen3-TTS. Use for narration/presentation before create_video with infinitalk mode.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The narration script to convert to speech"},
                        "language": {"type": "string", "enum": ["Korean", "English", "Chinese", "Japanese"]},
                        "speaker": {"type": "string", "description": "Speaker name, e.g. Ryan, Claire, Laura"},
                    },
                    "required": ["text", "language"],
                },
            },
            {
                "name": "list_gallery_images",
                "description": "List available images from outputs (FLUX-generated) and uploads.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "enum": ["outputs", "uploads", "all"]},
                    },
                },
            },
        ],
    },
]

STUDIO_SYSTEM_PROMPT = """You are IdolMotion, a versatile video creation assistant.
You help users create high-quality videos across multiple content types:

SUPPORTED CONTENT TYPES:
1. Dance — K-Pop idol dance, choreography. FFLF looping (33f, portrait) or Change Character.
2. Narration — Storytelling, lectures. InfiniTalk: single image + audio → unlimited video (landscape).
3. Presentation — Product showcase, slides. FFLF (49f, landscape) or InfiniTalk.

WORKFLOW:
1. Identify content type from user's request
2. Search and select appropriate LoRA adapters
3. Generate consistent images using FLUX (same seed)
4. For narration: generate TTS audio first, then InfiniTalk video
5. Create video with appropriate mode

RULES:
- Same seed for all images (character consistency)
- FFLF: 2-4 keyframe images. InfiniTalk: 1 image + audio. Change Character: 1 image + ref video.
- Default aspect: dance=portrait, narration/presentation=landscape
- Looping: dance=ON, others=OFF
- InfiniTalk: 832×480, 20 FPS, frames = audio_duration × 20
- For narration: ALWAYS generate_tts first, then create_video infinitalk
- Respond in the user's language
- Use trigger words from LoRA metadata"""


class ChatRequest(BaseModel):
    message: str
    history: list = []


@app.post("/api/studio/chat")
async def studio_chat(request: ChatRequest, user=Depends(get_current_user)):
    """Gemini function-calling chat for Video Studio AI assistant."""
    import google.generativeai as genai

    gemini_key = os.environ.get("GEMINI_KEY", "")
    if not gemini_key:
        raise HTTPException(503, "GEMINI_KEY not configured")

    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-pro",
        tools=STUDIO_TOOLS,
        system_instruction=STUDIO_SYSTEM_PROMPT,
    )
    chat = model.start_chat(history=request.history)
    response = chat.send_message(request.message)

    actions = []
    tool_calls = []

    # Function calling loop (multi-turn)
    max_rounds = 10
    for _ in range(max_rounds):
        has_function_call = False
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                has_function_call = True
                fc = part.function_call
                tool_calls.append(fc.name)
                result = await _execute_studio_tool(fc.name, dict(fc.args))
                actions.append({"tool": fc.name, "result": result})
                response = chat.send_message(
                    genai.types.ContentDict(
                        role="function",
                        parts=[genai.types.PartDict(
                            function_response=genai.types.FunctionResponseDict(
                                name=fc.name, response=result,
                            )
                        )]
                    )
                )
                break
        if not has_function_call:
            break

    reply = ""
    for part in response.candidates[0].content.parts:
        if hasattr(part, 'text') and part.text:
            reply += part.text

    return {"reply": reply, "actions": actions, "tool_calls": tool_calls}


async def _execute_studio_tool(name: str, args: dict) -> dict:
    """Execute a Gemini function call tool."""
    if name == "search_lora":
        query = args["query"].lower()
        category = args.get("category")
        matches = []
        for a in LORA_ADAPTERS:
            if category and a.get("category") != category:
                continue
            searchable = " ".join([
                a["name"].lower(), a.get("description", "").lower(),
                " ".join(a.get("trigger_words", [])).lower(),
                " ".join(a.get("semantic_tags", [])).lower(),
            ])
            if any(word in searchable for word in query.split()):
                matches.append({
                    "name": a["name"], "type": a["type"],
                    "trigger_words": a.get("trigger_words", []),
                    "recommended_weight": a.get("default_high_weight", 0.5),
                    "description": a.get("description", ""),
                })
        return {"matches": matches, "count": len(matches)}

    elif name == "generate_idol_image":
        task_id = str(uuid.uuid4())
        generation_status[task_id] = {
            "status": "pending", "progress": 0,
            "message": "Queued", "output_path": None,
        }
        executor.submit(generate_flux_task, task_id, {
            "prompt": args["prompt"],
            "seed": args.get("seed", -1),
            "aspect_ratio": args.get("aspect_ratio", "portrait"),
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "upscale": False,
            "lora_weights": args.get("lora_weights"),
        })
        # Wait for completion (max 120s)
        for _ in range(60):
            await asyncio.sleep(2)
            status = generation_status.get(task_id, {})
            if status.get("status") in ("completed", "failed"):
                break
        return generation_status.get(task_id, {})

    elif name == "create_video":
        mode = args.get("mode", "fflf")
        content_type = args.get("content_type", "dance")
        task_id = str(uuid.uuid4())
        generation_status[task_id] = {
            "status": "pending", "progress": 0,
            "message": "Queued", "output_path": None,
        }

        CONTENT_DEFAULTS = {
            "dance":        {"segment": 33, "looping": True,  "aspect": "portrait",  "prompt": "A beautiful idol dancing smoothly, dynamic lighting"},
            "narration":    {"segment": 65, "looping": False, "aspect": "landscape", "prompt": "A person speaking naturally, clear expression, professional lighting"},
            "presentation": {"segment": 49, "looping": False, "aspect": "landscape", "prompt": "Professional product showcase, smooth transition, clean background"},
        }
        defaults = CONTENT_DEFAULTS.get(content_type, CONTENT_DEFAULTS["dance"])

        if mode == "fflf":
            folder_result = await prepare_workflow_images({"image_paths": args["image_paths"]})
            seg_default = [defaults["segment"]] * len(args["image_paths"])
            segment_lengths = "\n".join(str(s) for s in args.get("segment_lengths", seg_default))
            executor.submit(workflow_generate_task, task_id, {
                "workflow_id": "fflf_auto_v2",
                "inputs": {
                    "images": folder_result["folder_path"],
                    "positive_prompt": args.get("master_prompt", defaults["prompt"]),
                    "segment_lengths": segment_lengths,
                    "seed": args.get("seed", 138),
                    "looping": args.get("looping", defaults["looping"]),
                },
            })
        elif mode == "infinitalk":
            audio_path = args.get("audio_path", "")
            if not audio_path:
                return {"error": "audio_path is required for infinitalk mode"}
            try:
                import librosa
                duration = librosa.get_duration(filename=audio_path)
                frame_count = int(duration * 20)
            except Exception:
                frame_count = 160
            executor.submit(workflow_generate_task, task_id, {
                "workflow_id": "wan_infinitalk",
                "inputs": {
                    "image": args["image_paths"][0],
                    "audio": audio_path,
                    "prompt": args.get("master_prompt", defaults["prompt"]),
                    "length": frame_count,
                },
            })
        elif mode == "change_character":
            ref_video_path = args.get("ref_video_url", "")
            if ref_video_path.startswith("http"):
                dl_result = download_youtube_video(ref_video_path)
                ref_video_path = dl_result["path"]
            executor.submit(workflow_generate_task, task_id, {
                "workflow_id": "change_character",
                "inputs": {
                    "ref_image": args["image_paths"][0],
                    "ref_video": ref_video_path,
                    "prompt": args.get("master_prompt", "The character is dancing"),
                    "aspect_ratio": args.get("aspect_ratio", defaults["aspect"]),
                },
            })
        return {"task_id": task_id, "status": "started", "mode": mode, "content_type": content_type}

    elif name == "generate_tts":
        import soundfile as sf
        try:
            model = get_tts_model()
            wavs, sr = model.generate_custom_voice(
                text=args["text"],
                language=args.get("language", "Korean"),
                speaker=args.get("speaker", "Ryan"),
            )
            filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
            audio_dir = UPLOAD_DIR / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio_path = str(audio_dir / filename)
            sf.write(audio_path, wavs[0], sr)
            duration = len(wavs[0]) / sr
            return {
                "audio_path": audio_path,
                "audio_url": f"/uploads/audio/{filename}",
                "duration": round(duration, 2),
                "frame_count": int(duration * 20),
            }
        except Exception as e:
            return {"error": f"TTS failed: {str(e)}"}

    elif name == "list_gallery_images":
        result = await list_outputs()
        images = [o for o in result["outputs"] if o["type"] == "image"]
        return {"images": images[:20]}

    return {"error": f"Unknown tool: {name}"}


# ============================================================================
# Catch-all: serve static files and SPA fallback (must be LAST route)
# ============================================================================
@app.get("/{filename:path}")
async def serve_static(filename: str):
    """Serve static files from frontend dist (logo, favicon, etc.)."""
    file_path = FRONTEND_DIR / filename
    if file_path.is_file() and FRONTEND_DIR in file_path.resolve().parents:
        return FileResponse(file_path)
    # SPA fallback: return index.html for client-side routing
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Not found")


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
    logging.info(f"I2V GGUF: {I2V_GGUF_DIR} (high={os.path.exists(I2V_GGUF_HIGH)}, low={os.path.exists(I2V_GGUF_LOW)})")
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
