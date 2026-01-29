#!/usr/bin/env python3
"""
WanAvatar - Speech-to-Video Generation App
Based on Wan2.2-S2V-14B model
"""
import os
import sys
import gc
import argparse
import datetime
import random
import logging
import shutil

# Add Wan2.2 to path
sys.path.insert(0, '/home/work/Wan2.2')

import torch
import numpy as np
import gradio as gr
from PIL import Image

# Wan2.2 imports
import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS

# Optional imports
try:
    from moviepy.editor import VideoFileClip
    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False
    print("Warning: moviepy not installed. Audio extraction will be disabled.")

try:
    from audio_separator.separator import Separator
    HAS_AUDIO_SEPARATOR = True
except ImportError:
    HAS_AUDIO_SEPARATOR = False
    print("Warning: audio-separator not installed. Vocal separation will be disabled.")


# ============================================================================
# Configuration
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--server_name", type=str, default="0.0.0.0", help="Server IP address")
parser.add_argument("--server_port", type=int, default=7891, help="Server port")
parser.add_argument("--share", action="store_true", help="Enable Gradio share")
parser.add_argument("--offload", action="store_true", default=True, help="Enable model offload to save VRAM")
args = parser.parse_args()

# Model paths
CHECKPOINT_DIR = "/home/work/Wan2.2/Wan2.2-S2V-14B"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device setup
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

# Global pipeline (lazy loading)
pipeline = None


# ============================================================================
# Model Loading
# ============================================================================
def load_pipeline():
    """Load Wan2.2 S2V pipeline with lazy initialization."""
    global pipeline
    if pipeline is not None:
        return pipeline

    logging.info("Loading Wan2.2-S2V-14B model...")
    cfg = WAN_CONFIGS['s2v-14B']

    pipeline = wan.WanS2V(
        config=cfg,
        checkpoint_dir=CHECKPOINT_DIR,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,  # Save VRAM
        init_on_cpu=True,  # Load model to CPU first
    )

    logging.info("Model loaded successfully!")
    return pipeline


# ============================================================================
# Generation Functions
# ============================================================================
def generate_video(
    image_path,
    audio_path,
    prompt,
    negative_prompt,
    resolution,
    num_clips,
    inference_steps,
    guidance_scale,
    infer_frames,
    seed,
    offload_model,
    progress=gr.Progress()
):
    """Generate talking head video from image and audio."""
    global pipeline

    if image_path is None:
        return None, -1, "Error: Please upload an image / 오류: 이미지를 업로드하세요"
    if audio_path is None:
        return None, -1, "Error: Please upload audio / 오류: 오디오를 업로드하세요"

    try:
        progress(0.1, desc="Loading model...")
        pipeline = load_pipeline()

        # Parse seed
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        # Parse resolution
        max_area = MAX_AREA_CONFIGS.get(resolution, 720 * 1280)

        # Log parameters
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.info(f"\n{'='*60}")
        logging.info(f"Generation started at {timestamp}")
        logging.info(f"Resolution: {resolution}, Max Area: {max_area}")
        logging.info(f"Inference Steps: {inference_steps}, Guidance Scale: {guidance_scale}")
        logging.info(f"Infer Frames: {infer_frames}, Num Clips: {num_clips}")
        logging.info(f"Seed: {seed}")
        logging.info(f"{'='*60}\n")

        progress(0.2, desc="Generating video...")

        # Generate
        video = pipeline.generate(
            input_prompt=prompt,
            ref_image_path=image_path,
            audio_path=audio_path,
            enable_tts=False,
            tts_prompt_audio=None,
            tts_prompt_text=None,
            tts_text=None,
            num_repeat=num_clips if num_clips > 0 else None,
            pose_video=None,
            max_area=max_area,
            infer_frames=infer_frames,
            shift=3.0,  # Recommended for S2V
            sample_solver='unipc',
            sampling_steps=inference_steps,
            guide_scale=guidance_scale,
            n_prompt=negative_prompt,
            seed=seed,
            offload_model=offload_model,
            init_first_frame=False,
        )

        progress(0.8, desc="Saving video...")

        # Save video
        from wan.utils.utils import save_video, merge_video_audio

        video_path = os.path.join(OUTPUT_DIR, f"{timestamp}.mp4")
        save_video(
            tensor=video[None],
            save_file=video_path,
            fps=16,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )

        # Merge audio
        output_path = os.path.join(OUTPUT_DIR, f"{timestamp}_with_audio.mp4")
        merge_video_audio(video_path=video_path, audio_path=audio_path, output_path=output_path)

        # Cleanup
        del video
        gc.collect()
        torch.cuda.empty_cache()

        progress(1.0, desc="Complete!")
        return output_path, seed, f"Success! Saved to {output_path}"

    except Exception as e:
        logging.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, -1, f"Error: {str(e)}"


def extract_audio(video_path):
    """Extract audio from video file."""
    if not HAS_MOVIEPY:
        return None, "Error: moviepy not installed"
    if video_path is None:
        return None, "Error: Please upload a video / 오류: 비디오를 업로드하세요"

    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"{timestamp}_extracted.wav")

        video = VideoFileClip(video_path)
        video.audio.write_audiofile(output_path, codec='pcm_s16le')
        video.close()

        return output_path, f"Success! Saved to {output_path}"
    except Exception as e:
        return None, f"Error: {str(e)}"


def separate_vocals(audio_path):
    """Separate vocals from audio file."""
    if not HAS_AUDIO_SEPARATOR:
        return None, "Error: audio-separator not installed"
    if audio_path is None:
        return None, "Error: Please upload audio / 오류: 오디오를 업로드하세요"

    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = os.path.join(OUTPUT_DIR, f"temp_{timestamp}")
        os.makedirs(temp_dir, exist_ok=True)

        separator = Separator(
            output_dir=temp_dir,
            output_single_stem="vocals",
        )
        separator.load_model("Kim_Vocal_2.onnx")
        outputs = separator.separate(audio_path)

        vocal_file = os.path.join(temp_dir, outputs[0])
        output_path = os.path.join(OUTPUT_DIR, f"{timestamp}_vocals.wav")

        shutil.move(vocal_file, output_path)
        shutil.rmtree(temp_dir, ignore_errors=True)

        return output_path, f"Success! Saved to {output_path}"
    except Exception as e:
        return None, f"Error: {str(e)}"


# ============================================================================
# UI Translations
# ============================================================================
TRANSLATIONS = {
    "en": {
        "title": "WanAvatar - Talking Head Generation",
        "subtitle": "Generate realistic talking head videos from a single image and audio",
        "tab_generate": "Generate Video",
        "tab_extract": "Extract Audio",
        "tab_separate": "Vocal Separation",
        "image_label": "Reference Image",
        "audio_label": "Driving Audio",
        "prompt_label": "Prompt",
        "prompt_default": "A person speaking naturally with subtle expressions, minimal head movement, simple blinking, neutral background",
        "neg_prompt_label": "Negative Prompt",
        "neg_prompt_default": "blurry, low quality, distorted face, unnatural movements, artifacts, text, watermark",
        "resolution_label": "Resolution",
        "num_clips_label": "Number of Clips (0=auto)",
        "steps_label": "Inference Steps",
        "guidance_label": "Guidance Scale",
        "frames_label": "Frames per Clip",
        "seed_label": "Seed (-1=random)",
        "offload_label": "Enable Model Offload (saves VRAM)",
        "generate_btn": "Generate Video",
        "video_output": "Generated Video",
        "seed_output": "Used Seed",
        "status": "Status",
        "video_input": "Upload Video",
        "extract_btn": "Extract Audio",
        "audio_output": "Extracted Audio",
        "audio_input": "Upload Audio",
        "separate_btn": "Separate Vocals",
        "vocals_output": "Separated Vocals",
    },
    "ko": {
        "title": "WanAvatar - 토킹 헤드 생성",
        "subtitle": "단일 이미지와 오디오로 사실적인 토킹 헤드 비디오 생성",
        "tab_generate": "비디오 생성",
        "tab_extract": "오디오 추출",
        "tab_separate": "보컬 분리",
        "image_label": "참조 이미지",
        "audio_label": "구동 오디오",
        "prompt_label": "프롬프트",
        "prompt_default": "자연스럽게 말하는 사람, 미세한 표정, 최소한의 머리 움직임, 단순한 눈 깜빡임, 중립적 배경",
        "neg_prompt_label": "네거티브 프롬프트",
        "neg_prompt_default": "흐림, 저화질, 왜곡된 얼굴, 부자연스러운 움직임, 아티팩트, 텍스트, 워터마크",
        "resolution_label": "해상도",
        "num_clips_label": "클립 수 (0=자동)",
        "steps_label": "추론 스텝",
        "guidance_label": "가이던스 스케일",
        "frames_label": "클립당 프레임",
        "seed_label": "시드 (-1=랜덤)",
        "offload_label": "모델 오프로드 활성화 (VRAM 절약)",
        "generate_btn": "비디오 생성",
        "video_output": "생성된 비디오",
        "seed_output": "사용된 시드",
        "status": "상태",
        "video_input": "비디오 업로드",
        "extract_btn": "오디오 추출",
        "audio_output": "추출된 오디오",
        "audio_input": "오디오 업로드",
        "separate_btn": "보컬 분리",
        "vocals_output": "분리된 보컬",
    },
    "zh": {
        "title": "WanAvatar - 说话人头生成",
        "subtitle": "从单张图片和音频生成逼真的说话人头视频",
        "tab_generate": "生成视频",
        "tab_extract": "提取音频",
        "tab_separate": "人声分离",
        "image_label": "参考图片",
        "audio_label": "驱动音频",
        "prompt_label": "提示词",
        "prompt_default": "一个人自然说话，细微表情，头部动作轻微，简单眨眼，中性背景",
        "neg_prompt_label": "负面提示词",
        "neg_prompt_default": "模糊，低质量，扭曲的脸，不自然的动作，伪影，文字，水印",
        "resolution_label": "分辨率",
        "num_clips_label": "片段数量（0=自动）",
        "steps_label": "推理步数",
        "guidance_label": "引导比例",
        "frames_label": "每片段帧数",
        "seed_label": "种子（-1=随机）",
        "offload_label": "启用模型卸载（节省显存）",
        "generate_btn": "生成视频",
        "video_output": "生成的视频",
        "seed_output": "使用的种子",
        "status": "状态",
        "video_input": "上传视频",
        "extract_btn": "提取音频",
        "audio_output": "提取的音频",
        "audio_input": "上传音频",
        "separate_btn": "分离人声",
        "vocals_output": "分离的人声",
    }
}


def get_text(key, lang="en"):
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)


# ============================================================================
# Custom CSS for Modern UI
# ============================================================================
CUSTOM_CSS = """
/* Global Styles */
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 2rem !important;
}

/* Header Styles */
.header-container {
    text-align: center;
    padding: 2rem 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
}

.header-title {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}

.header-subtitle {
    font-size: 1.1rem;
    opacity: 0.9;
}

/* Card Styles */
.input-card, .output-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    border: 1px solid #e5e7eb;
}

/* Button Styles */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    font-size: 1.1rem !important;
    transition: all 0.3s ease !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
}

/* Tab Styles */
.tab-nav {
    border-bottom: 2px solid #e5e7eb;
    margin-bottom: 1rem;
}

.tab-nav button {
    font-weight: 600;
    padding: 0.75rem 1.5rem;
}

.tab-nav button.selected {
    border-bottom: 3px solid #667eea;
    color: #667eea;
}

/* Status Box */
.status-box {
    background: #f8fafc;
    border-radius: 8px;
    padding: 1rem;
    border-left: 4px solid #667eea;
}

/* Language Selector */
.language-selector {
    position: absolute;
    top: 1rem;
    right: 1rem;
}

/* Slider Improvements */
input[type="range"] {
    accent-color: #667eea;
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    .input-card, .output-card {
        background: #1f2937;
        border-color: #374151;
    }

    .status-box {
        background: #111827;
    }
}
"""


# ============================================================================
# Gradio Interface
# ============================================================================
def create_interface():
    """Create the Gradio interface."""

    with gr.Blocks(
        title="WanAvatar",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
        )
    ) as demo:

        # State for language
        current_lang = gr.State("en")

        # Header
        gr.HTML("""
        <div class="header-container">
            <div class="header-title">WanAvatar</div>
            <div class="header-subtitle">Talking Head Video Generation powered by Wan2.2-S2V-14B</div>
        </div>
        """)

        # Language selector
        with gr.Row():
            language = gr.Radio(
                choices=[("English", "en"), ("한국어", "ko"), ("中文", "zh")],
                value="en",
                label="Language / 언어 / 语言",
                scale=1
            )

        # Main tabs
        with gr.Tabs() as tabs:

            # ==================== Generate Tab ====================
            with gr.TabItem("Generate Video", id="generate"):
                with gr.Row():
                    # Left column - Inputs
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### Input")

                            image_input = gr.Image(
                                label="Reference Image",
                                type="filepath",
                                height=300,
                            )

                            audio_input = gr.Audio(
                                label="Driving Audio",
                                type="filepath",
                            )

                        with gr.Accordion("Prompt Settings", open=False):
                            prompt = gr.Textbox(
                                label="Prompt",
                                value="A person speaking naturally with subtle expressions, minimal head movement, simple blinking, neutral background",
                                lines=3,
                            )
                            neg_prompt = gr.Textbox(
                                label="Negative Prompt",
                                value="blurry, low quality, distorted face, unnatural movements, artifacts, text, watermark",
                                lines=2,
                            )

                        with gr.Accordion("Generation Settings", open=True):
                            resolution = gr.Dropdown(
                                label="Resolution",
                                choices=list(SIZE_CONFIGS.keys()),
                                value="720*1280",
                            )

                            with gr.Row():
                                num_clips = gr.Slider(
                                    label="Number of Clips (0=auto)",
                                    minimum=0,
                                    maximum=10,
                                    step=1,
                                    value=0,
                                )
                                infer_frames = gr.Slider(
                                    label="Frames per Clip",
                                    minimum=48,
                                    maximum=120,
                                    step=4,
                                    value=80,
                                )

                            with gr.Row():
                                inference_steps = gr.Slider(
                                    label="Inference Steps",
                                    minimum=20,
                                    maximum=100,
                                    step=5,
                                    value=40,
                                )
                                guidance_scale = gr.Slider(
                                    label="Guidance Scale",
                                    minimum=1.0,
                                    maximum=10.0,
                                    step=0.5,
                                    value=4.5,
                                )

                            with gr.Row():
                                seed = gr.Number(
                                    label="Seed (-1=random)",
                                    value=-1,
                                    precision=0,
                                )
                                offload = gr.Checkbox(
                                    label="Model Offload (saves VRAM)",
                                    value=True,
                                )

                        generate_btn = gr.Button(
                            "Generate Video",
                            variant="primary",
                            size="lg",
                        )

                    # Right column - Outputs
                    with gr.Column(scale=1):
                        gr.Markdown("### Output")

                        video_output = gr.Video(
                            label="Generated Video",
                            height=400,
                        )

                        with gr.Row():
                            seed_output = gr.Textbox(
                                label="Used Seed",
                                interactive=False,
                            )

                        status_output = gr.Textbox(
                            label="Status",
                            interactive=False,
                        )

                # Event handlers
                generate_btn.click(
                    fn=generate_video,
                    inputs=[
                        image_input, audio_input, prompt, neg_prompt,
                        resolution, num_clips, inference_steps, guidance_scale,
                        infer_frames, seed, offload
                    ],
                    outputs=[video_output, seed_output, status_output],
                )

            # ==================== Extract Audio Tab ====================
            with gr.TabItem("Extract Audio", id="extract"):
                with gr.Row():
                    with gr.Column():
                        video_for_extract = gr.Video(
                            label="Upload Video",
                            height=350,
                        )
                        extract_btn = gr.Button(
                            "Extract Audio",
                            variant="primary",
                        )

                    with gr.Column():
                        extracted_audio = gr.Audio(
                            label="Extracted Audio",
                        )
                        extract_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                        )

                extract_btn.click(
                    fn=extract_audio,
                    inputs=[video_for_extract],
                    outputs=[extracted_audio, extract_status],
                )

            # ==================== Vocal Separation Tab ====================
            with gr.TabItem("Vocal Separation", id="separate"):
                with gr.Row():
                    with gr.Column():
                        audio_for_separate = gr.Audio(
                            label="Upload Audio",
                            type="filepath",
                        )
                        separate_btn = gr.Button(
                            "Separate Vocals",
                            variant="primary",
                        )

                    with gr.Column():
                        separated_vocals = gr.Audio(
                            label="Separated Vocals",
                        )
                        separate_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                        )

                separate_btn.click(
                    fn=separate_vocals,
                    inputs=[audio_for_separate],
                    outputs=[separated_vocals, separate_status],
                )

        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 1rem; margin-top: 2rem; color: #6b7280; font-size: 0.9rem;">
            <p>Powered by Wan2.2-S2V-14B | WanAvatar</p>
        </div>
        """)

        # Language change handler
        def update_ui_language(lang):
            t = lambda k: get_text(k, lang)
            return {
                image_input: gr.Image(label=t("image_label")),
                audio_input: gr.Audio(label=t("audio_label")),
                prompt: gr.Textbox(label=t("prompt_label"), value=t("prompt_default")),
                neg_prompt: gr.Textbox(label=t("neg_prompt_label"), value=t("neg_prompt_default")),
                resolution: gr.Dropdown(label=t("resolution_label")),
                num_clips: gr.Slider(label=t("num_clips_label")),
                inference_steps: gr.Slider(label=t("steps_label")),
                guidance_scale: gr.Slider(label=t("guidance_label")),
                infer_frames: gr.Slider(label=t("frames_label")),
                seed: gr.Number(label=t("seed_label")),
                offload: gr.Checkbox(label=t("offload_label")),
                generate_btn: gr.Button(t("generate_btn")),
                video_output: gr.Video(label=t("video_output")),
                seed_output: gr.Textbox(label=t("seed_output")),
                status_output: gr.Textbox(label=t("status")),
                video_for_extract: gr.Video(label=t("video_input")),
                extract_btn: gr.Button(t("extract_btn")),
                extracted_audio: gr.Audio(label=t("audio_output")),
                extract_status: gr.Textbox(label=t("status")),
                audio_for_separate: gr.Audio(label=t("audio_input")),
                separate_btn: gr.Button(t("separate_btn")),
                separated_vocals: gr.Audio(label=t("vocals_output")),
                separate_status: gr.Textbox(label=t("status")),
            }

        language.change(
            fn=update_ui_language,
            inputs=[language],
            outputs=[
                image_input, audio_input, prompt, neg_prompt,
                resolution, num_clips, inference_steps, guidance_scale,
                infer_frames, seed, offload, generate_btn,
                video_output, seed_output, status_output,
                video_for_extract, extract_btn, extracted_audio, extract_status,
                audio_for_separate, separate_btn, separated_vocals, separate_status,
            ],
        )

    return demo


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    logging.info("Starting WanAvatar...")
    logging.info(f"Checkpoint directory: {CHECKPOINT_DIR}")
    logging.info(f"Device: {DEVICE}, Dtype: {DTYPE}")

    demo = create_interface()
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=True,
    )
