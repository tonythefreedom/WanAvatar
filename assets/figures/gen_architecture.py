#!/usr/bin/env python3
"""Generate CineSynth AI architecture diagram."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(22, 16), dpi=150)
ax.set_xlim(0, 22)
ax.set_ylim(0, 16)
ax.axis('off')
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

# Color palette
C = {
    'bg': '#0d1117', 'card': '#161b22', 'border': '#30363d',
    'primary': '#58a6ff', 'green': '#3fb950', 'purple': '#bc8cff',
    'orange': '#d29922', 'red': '#f85149', 'pink': '#f778ba',
    'text': '#e6edf3', 'text2': '#8b949e', 'white': '#ffffff',
    'gpu0': '#1f6feb', 'gpu1': '#8957e5',
}

def draw_box(x, y, w, h, color, title, items=None, title_color=None, radius=0.15):
    rect = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.05,rounding_size={radius}",
                          facecolor=color, edgecolor=C['border'], linewidth=1.2, zorder=2)
    ax.add_patch(rect)
    tc = title_color or C['white']
    ax.text(x + w/2, y + h - 0.28, title, ha='center', va='top',
            fontsize=9, fontweight='bold', color=tc, zorder=3)
    if items:
        for i, item in enumerate(items):
            ax.text(x + 0.15, y + h - 0.58 - i*0.28, item, ha='left', va='top',
                    fontsize=6.5, color=C['text2'], zorder=3, fontfamily='monospace')

def draw_arrow(x1, y1, x2, y2, color=C['primary'], style='->', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, connectionstyle='arc3,rad=0.05'),
                zorder=4)

def draw_section(x, y, w, h, title, color):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor='none', edgecolor=color, linewidth=1.8, linestyle='--', zorder=1)
    ax.add_patch(rect)
    ax.text(x + 0.15, y + h - 0.12, title, ha='left', va='top',
            fontsize=8, fontweight='bold', color=color, zorder=3,
            bbox=dict(boxstyle='round,pad=0.2', facecolor=C['bg'], edgecolor='none'))

# ===== TITLE =====
ax.text(11, 15.6, 'CineSynth AI', ha='center', va='center',
        fontsize=24, fontweight='bold', color=C['primary'], zorder=5)
ax.text(11, 15.2, 'AI Video Production Studio — System Architecture',
        ha='center', va='center', fontsize=10, color=C['text2'], zorder=5)

# ===== FRONTEND SECTION =====
draw_section(0.3, 11.5, 9.4, 3.3, 'FRONTEND  (React + Vite)', C['green'])

draw_box(0.6, 13.1, 1.7, 1.4, '#1a2332', 'Video Studio',
         ['Manual Mode', 'AI Chat Mode', 'Timeline DnD', 'Content Presets'], C['green'])
draw_box(2.5, 13.1, 1.5, 1.4, '#1a2332', 'Image Gen',
         ['FLUX Prompt', 'LoRA Select', 'Upscale x2', 'Aspect Ratio'], C['green'])
draw_box(4.2, 13.1, 1.5, 1.4, '#1a2332', 'Video Gen',
         ['I2V Pipeline', 'Motion LoRA', 'Frame Control', 'Shift/CFG'], C['green'])
draw_box(5.9, 13.1, 1.5, 1.4, '#1a2332', 'Workflow',
         ['ComfyUI', 'Change Char', 'FFLF Loop', 'InfiniTalk'], C['green'])
draw_box(7.6, 13.1, 1.8, 1.4, '#1a2332', 'Gallery',
         ['Images Tab', 'Videos Tab', 'Multi-Select', 'Delete/Preview'], C['green'])

# API layer
draw_box(0.6, 11.7, 8.8, 1.1, '#121a24', 'api.js — Axios HTTP Client  (30+ endpoints)',
         ['POST /api/generate*  |  GET /api/status/{id}  |  POST /api/upload/*  |  POST /api/studio/chat  |  GET /api/workflows'],
         C['text2'])

# ===== BACKEND SECTION =====
draw_section(0.3, 3.8, 9.4, 7.4, 'BACKEND  (FastAPI + Python 3.12)', C['primary'])

# Server core
draw_box(0.6, 9.3, 4.2, 1.7, '#1a2332', 'server.py — FastAPI Application',
         ['ThreadPoolExecutor (2 workers)', 'GPU Lock Manager (gpu0_lock, gpu1_lock)',
          'generation_status{} — Task Tracker', 'Model Swap: ensure_model_loaded()',
          'Static Files: /outputs, /uploads, /assets'], C['primary'])

draw_box(5.0, 9.3, 4.4, 1.7, '#1a2332', 'Studio AI  (Solar-Banya-100b)',
         ['Function Calling (5 tools)', 'search_lora → LoRA Discovery',
          'generate_idol_image → FLUX', 'create_video → FFLF/InfiniTalk/ChangeChar',
          'generate_tts → Qwen3-TTS'], C['purple'])

# Model pipelines
draw_box(0.6, 7.2, 2.2, 1.8, '#0d2137', 'S2V Pipeline',
         ['Wan2.2-S2V-14B', 'Sequence Parallel', 'Multi-LoRA MoE', 'AutoRegressive', 'Audio→Lipsync'], C['gpu0'])

draw_box(3.0, 7.2, 2.0, 1.8, '#0d2137', 'I2V Pipeline',
         ['Wan2.2-I2V-14B', 'GGUF Q4_K_M', 'HiNoise+LoNoise', 'Multi-LoRA MoE', 'Image→Video'], C['gpu0'])

draw_box(5.2, 7.2, 1.8, 1.8, '#1a1a2e', 'FLUX Pipeline',
         ['FLUX.2-klein-9B', '4-step Fast Gen', 'Real-ESRGAN x2', 'Text→Image'], C['gpu1'])

draw_box(7.2, 7.2, 2.2, 1.8, '#1a1a2e', 'TTS Pipeline',
         ['Qwen3-TTS-0.6B', 'CustomVoice', '10 Speakers', '10 Languages', 'Text→Speech'], C['gpu1'])

# LoRA system
draw_box(0.6, 5.5, 4.5, 1.4, '#1a2218', 'LoRA Adapter System',
         ['KoreanWoman (char)  |  UlzzangG1 (char)  |  UkaSexyLight (motion)',
          'HipSway (motion)  |  OrbitCamV2 (camera)  |  Auto-discovery /lora_adpts/',
          'Semantic Tags for AI Search  |  High/Low Noise MoE Strategy'], C['green'])

draw_box(5.3, 5.5, 4.1, 1.4, '#221a1a', 'ComfyUI Integration',
         ['SetNode/GetNode Resolution', 'WebSocket Progress Monitor',
          'File Upload/Download', 'Audio Merge (ffmpeg)'], C['orange'])

# Workflow registry
draw_box(0.6, 4.0, 3.0, 1.2, '#1a1520', 'Change Character V1.1',
         ['Pose Detection + SAM', 'WanAnimate 14B', 'Character Swap'], C['pink'])
draw_box(3.8, 4.0, 2.8, 1.2, '#1a1520', 'InfiniTalk',
         ['Image + Audio', 'Unlimited Duration', 'wav2vec2 Guide'], C['pink'])
draw_box(6.8, 4.0, 2.6, 1.2, '#1a1520', 'FFLF Auto V2',
         ['Image Sequence', 'Looping Video', 'GIMMVFI Interp'], C['pink'])

# ===== GPU SECTION =====
draw_section(10.2, 3.8, 5.5, 7.4, 'GPU MEMORY  (2x A100 80GB)', C['orange'])

# GPU 0
draw_box(10.5, 8.6, 4.9, 2.3, '#0d1a2e', 'cuda:0 — GPU 0  (80 GB)',
         ['', ''], C['gpu0'])
draw_box(10.7, 8.8, 2.2, 1.5, '#0a1628', 'S2V (Exclusive)',
         ['DiT 14B: ~28GB', 'T5+VAE+Audio', 'Total: ~39GB'], '#4a8fe7')
draw_box(13.1, 8.8, 2.1, 1.5, '#0a1628', 'I2V (Shared)',
         ['GGUF Q4_K_M', 'Hi+Lo Noise', 'Total: ~20GB'], '#4a8fe7')

# GPU 1
draw_box(10.5, 5.8, 4.9, 2.5, '#1a0d2e', 'cuda:1 — GPU 1  (80 GB)',
         ['', ''], C['gpu1'])
draw_box(10.7, 6.0, 2.2, 1.5, '#140a28', 'FLUX (Shared)',
         ['Klein-9B: ~18GB', '4-step Distill', 'Upscale: ~1GB'], '#9a6fdb')
draw_box(13.1, 6.0, 2.1, 1.5, '#140a28', 'TTS (Shared)',
         ['Qwen3: ~1GB', 'CustomVoice', 'Low Priority'], '#9a6fdb')

# Block swap
draw_box(10.5, 4.1, 4.9, 1.4, '#1a1a0d', 'Model Swap Manager',
         ['S2V needs both GPUs → Offloads I2V & FLUX to CPU',
          'I2V on cuda:0 ↔ FLUX on cuda:1 (concurrent OK)',
          'ComfyUI: Block Swap (12 blocks) for 14B Animate'], C['orange'])

# ===== STORAGE SECTION =====
draw_section(10.2, 0.3, 11.3, 3.2, 'STORAGE', C['text2'])

draw_box(10.5, 0.6, 3.5, 2.6, '#161b22', '/mnt/models/  (500GB disk)',
         ['Wan2.2-S2V-14B ......... 28GB', 'Wan2.2-I2V-GGUF ........ 11GB',
          'FLUX.2-klein-9B ........ 18GB', 'ComfyUI Models ......... 53GB',
          '  wan2.2_animate_14B ... 33GB', '  t5_uncensored ........ 11GB',
          '  5x LoRAs ............. 8GB', 'Total: ~121GB'], C['text2'])

draw_box(14.2, 0.6, 3.5, 2.6, '#161b22', 'Project Files',
         ['server.py ........... 2,512 lines', 'App.jsx ............. 2,713 lines',
          'api.js .............. 161 lines', 'workflow/ ........... 3 JSON files',
          'lora_adpts/ ......... 5 adapters', 'uploads/ ............ User files',
          'outputs/ ............ Generated', 'wan/ ................ Core ML code'], C['text2'])

draw_box(17.9, 0.6, 3.3, 2.6, '#161b22', 'External Services',
         ['ComfyUI .... localhost:8188', 'Solar-Banya ... AI Chat',
          'HuggingFace .. Model DL', 'YouTube-DL ... Video DL',
          'FFmpeg ....... A/V Merge', '', ''], C['text2'])

# ===== ARROWS =====
# Frontend → Backend
draw_arrow(5, 11.7, 5, 11.0, C['green'], '->')
ax.text(5.2, 11.35, 'HTTP/REST', fontsize=6, color=C['text2'], zorder=5)

# Server → Pipelines
draw_arrow(2.7, 9.3, 1.7, 9.0, C['gpu0'])
draw_arrow(3.5, 9.3, 4.0, 9.0, C['gpu0'])
draw_arrow(4.5, 9.3, 6.1, 9.0, C['gpu1'])
draw_arrow(5.0, 9.3, 8.3, 9.0, C['gpu1'])

# Studio → Pipelines
draw_arrow(6.0, 9.3, 6.1, 9.0, C['purple'])
draw_arrow(7.0, 9.3, 8.3, 9.0, C['purple'])

# Pipelines → GPU
draw_arrow(1.7, 7.2, 12.0, 7.15, C['gpu0'], '->', 1.0)
draw_arrow(4.0, 7.2, 13.6, 7.15, C['gpu0'], '->', 1.0)
draw_arrow(6.1, 7.2, 11.8, 7.15, C['gpu1'], '->', 1.0)
draw_arrow(8.3, 7.2, 14.2, 7.15, C['gpu1'], '->', 1.0)

# LoRA → Pipelines
draw_arrow(2.8, 6.9, 1.7, 7.2, C['green'], '->', 1.0)
draw_arrow(3.0, 6.9, 4.0, 7.2, C['green'], '->', 1.0)

# ComfyUI → Workflows
draw_arrow(7.3, 5.5, 2.1, 5.2, C['orange'], '->', 1.0)
draw_arrow(7.3, 5.5, 5.2, 5.2, C['orange'], '->', 1.0)
draw_arrow(7.3, 5.5, 8.1, 5.2, C['orange'], '->', 1.0)

# ===== DATA FLOW (right side) =====
draw_section(16.2, 3.8, 5.3, 11.0, 'DATA FLOW PIPELINES', C['pink'])

# Dance flow
draw_box(16.5, 12.5, 4.7, 2.0, '#1a1520', 'Dance Video Pipeline',
         ['1. FLUX → Pose Images (same seed)', '2. Timeline: 2-4 keyframes × 33f',
          '3a. FFLF → Looping dance video', '3b. ChangeChar → Ref video swap',
          '   Portrait 9:16, Loop ON'], C['pink'])

# Narration flow
draw_box(16.5, 10.1, 4.7, 2.1, '#1a1520', 'Narration Video Pipeline',
         ['1. FLUX → Character/Scene image', '2. Qwen3-TTS → WAV from script',
          '   or Upload audio file', '3. InfiniTalk → Unlimited video',
          '   Landscape 16:9, 20fps', '   Frame count = audio sec × 20'], C['pink'])

# Presentation flow
draw_box(16.5, 7.8, 4.7, 2.0, '#1a1520', 'Presentation Pipeline',
         ['1. FLUX → Slide/Product images', '2. Timeline: 2-6 images × 49f',
          '3a. FFLF → Transition video', '3b. TTS+InfiniTalk → Voiceover',
          '   Landscape 16:9, Loop OFF'], C['pink'])

# Change Character detail
draw_box(16.5, 5.4, 4.7, 2.1, '#1a1520', 'Change Character V1.1 Detail',
         ['1. Upload: Image + Ref Video (1080p)', '2. Pose Detection (ViTPose-L)',
          '3. Segmentation (SAM + DINO)', '4. WanAnimate 14B (4 steps, bf16)',
          '5. VAE Decode → Video Encode', '6. Audio Merge (ffmpeg)'], C['orange'])

# S2V Detail
draw_box(16.5, 4.0, 4.7, 1.1, '#0d2137', 'S2V Lipsync Detail',
         ['Audio → Wav2Vec2 → DiT 14B (SP)', 'LoRA MoE + Auto-Regressive segments',
          '25 steps, CFG 5.5, UniPC solver'], C['gpu0'])

# Legend
ax.text(0.5, 0.15, 'Legend:', fontsize=7, color=C['text'], fontweight='bold')
legend_items = [
    (C['gpu0'], 'cuda:0'), (C['gpu1'], 'cuda:1'), (C['green'], 'Frontend/LoRA'),
    (C['purple'], 'AI (Solar)'), (C['orange'], 'ComfyUI'), (C['pink'], 'Pipelines'),
]
for i, (color, label) in enumerate(legend_items):
    ax.add_patch(FancyBboxPatch((1.8 + i*1.4, 0.02), 0.3, 0.2,
                                boxstyle="round,pad=0.02", facecolor=color, edgecolor='none', zorder=5))
    ax.text(2.2 + i*1.4, 0.12, label, fontsize=6, color=C['text2'], va='center', zorder=5)

plt.tight_layout(pad=0.5)
plt.savefig('/home/ubuntu/WanAvatar/assets/figures/architecture.png',
            dpi=150, bbox_inches='tight', facecolor=C['bg'], edgecolor='none')
print("Saved: assets/figures/architecture.png")
