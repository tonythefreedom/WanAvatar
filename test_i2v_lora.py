#!/usr/bin/env python3
"""
Quick I2V LoRA test script — generates 17-frame (~1 sec) videos to verify LoRA effects.

Usage:
    source /mnt/models/wanavatarvenv/bin/activate
    python test_i2v_lora.py --test none       # Base model only (no LoRA)
    python test_i2v_lora.py --test ulzzang    # UlzzangG1 low noise only
    python test_i2v_lora.py --test uka        # UkaSexyLight high noise only
    python test_i2v_lora.py --test both       # UlzzangG1 + UkaSexyLight
    python test_i2v_lora.py --test hipsway   # HipSway high noise only
    python test_i2v_lora.py --test all        # Run all 5 tests sequentially
"""

import argparse
import gc
import logging
import os
import sys
import time

import torch

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)

# ── Paths ──
I2V_DIFFUSERS_DIR = "/mnt/models/Wan2.2-I2V-A14B-Diffusers"
I2V_GGUF_DIR = "/mnt/models/Wan2.2-I2V-A14B-GGUF"
I2V_GGUF_HIGH = os.path.join(I2V_GGUF_DIR, "HighNoise", "Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf")
I2V_GGUF_LOW = os.path.join(I2V_GGUF_DIR, "LowNoise", "Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf")

LORA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora_adpts")
ULZZANG_PATH = os.path.join(LORA_DIR, "mov", "character", "UlzzangG1.safetensors")
UKA_PATH = os.path.join(LORA_DIR, "mov", "move", "wan22-uka-sexy-light.safetensors")
HIPSWAY_PATH = os.path.join(LORA_DIR, "mov", "move", "wan22_i2v_zxtp_hip_sway_low_r1.safetensors")

IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "flux_20260211_054048.png")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# ── Generation config ──
PROMPT = "UlzzangG1, a beautiful young korean woman with large eyes, pale skin, makeup and small full lips, uka, A cinematic video with natural motion, high quality, smooth movement"
NEG_PROMPT = "ugly, blurry, low quality, distorted, deformed, static, frozen"
NUM_FRAMES = 17     # ~1 second at 16fps
HEIGHT = 1280
WIDTH = 720
STEPS = 20          # Fewer steps for quick testing
GUIDANCE = 5.0
SEED = 42           # Fixed seed for reproducibility


def load_pipeline():
    """Load GGUF I2V pipeline with both transformers."""
    from diffusers import WanImageToVideoPipeline, WanTransformer3DModel, GGUFQuantizationConfig

    logging.info("Loading GGUF Q4_K_M transformers...")
    qconfig = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)

    transformer_high = WanTransformer3DModel.from_single_file(
        I2V_GGUF_HIGH,
        config=I2V_DIFFUSERS_DIR,
        subfolder="transformer",
        quantization_config=qconfig,
        torch_dtype=torch.bfloat16,
    )
    transformer_low = WanTransformer3DModel.from_single_file(
        I2V_GGUF_LOW,
        config=I2V_DIFFUSERS_DIR,
        subfolder="transformer_2",
        quantization_config=qconfig,
        torch_dtype=torch.bfloat16,
    )

    logging.info("Loading pipeline (T5, CLIP, VAE, scheduler)...")
    pipe = WanImageToVideoPipeline.from_pretrained(
        I2V_DIFFUSERS_DIR,
        transformer=transformer_high,
        transformer_2=transformer_low,
        torch_dtype=torch.bfloat16,
    )

    # Load ALL LoRA adapters onto respective transformers
    logging.info("Loading UlzzangG1 onto transformer_2 (low noise)...")
    pipe.load_lora_weights(ULZZANG_PATH, adapter_name="UlzzangG1_low", load_into_transformer_2=True)

    logging.info("Loading UkaSexyLight onto transformer (high noise)...")
    pipe.load_lora_weights(UKA_PATH, adapter_name="UkaSexyLight_high")

    logging.info("Loading HipSway onto transformer (high noise)...")
    pipe.load_lora_weights(HIPSWAY_PATH, adapter_name="HipSway_high")

    pipe.to("cuda:0")
    logging.info("Pipeline loaded on cuda:0!")
    return pipe


def debug_peft_config(pipe):
    """Print PEFT adapter configuration for debugging."""
    logging.info("=" * 60)
    logging.info("PEFT ADAPTER DEBUG INFO")
    logging.info("=" * 60)

    for comp_name in ["transformer", "transformer_2"]:
        model = getattr(pipe, comp_name, None)
        if model is None:
            continue

        logging.info(f"\n--- {comp_name} ---")

        # List loaded adapters
        if hasattr(model, 'peft_config'):
            for adapter_name, config in model.peft_config.items():
                logging.info(f"  Adapter '{adapter_name}':")
                logging.info(f"    r (rank) = {config.r}")
                logging.info(f"    lora_alpha = {config.lora_alpha}")
                logging.info(f"    scaling = {config.lora_alpha / config.r:.6f}")
                logging.info(f"    target_modules = {config.target_modules}")

        # Active adapters
        if hasattr(model, 'active_adapters'):
            logging.info(f"  Active adapters: {model.active_adapters}")

        # Sample a LoRA module's scaling
        found = False
        for name, module in model.named_modules():
            if hasattr(module, 'scaling') and isinstance(module.scaling, dict):
                if not found:
                    for k, v in module.scaling.items():
                        logging.info(f"  Sample module '{name}' → scaling['{k}'] = {v}")
                    found = True
                    break

    logging.info("=" * 60)


def run_test(pipe, test_name, adapter_names=None, adapter_weights=None, prompt=None):
    """Run a single I2V generation test."""
    from PIL import Image
    from diffusers.utils import export_to_video

    prompt = prompt or PROMPT

    logging.info(f"\n{'='*60}")
    logging.info(f"TEST: {test_name}")
    logging.info(f"{'='*60}")

    # Step 1: Disable all LoRA
    pipe.disable_lora()
    logging.info("All LoRA disabled.")

    # Step 2: Enable selected adapters
    if adapter_names:
        pipe.enable_lora()
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
        logging.info(f"Activated: {list(zip(adapter_names, adapter_weights))}")

        # Verify active state
        for comp_name in ["transformer", "transformer_2"]:
            model = getattr(pipe, comp_name, None)
            if model and hasattr(model, 'active_adapters'):
                logging.info(f"  {comp_name} active: {model.active_adapters}")
    else:
        logging.info("Running without LoRA (base model).")

    # Step 3: Generate
    img = Image.open(IMAGE_PATH).convert("RGB")
    generator = torch.Generator(device="cpu").manual_seed(SEED)

    logging.info(f"Generating: {HEIGHT}x{WIDTH}, frames={NUM_FRAMES}, steps={STEPS}, seed={SEED}")
    t0 = time.time()

    logging.info(f"Prompt: {prompt}")
    output = pipe(
        image=img,
        prompt=prompt,
        negative_prompt=NEG_PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=generator,
    )

    elapsed = time.time() - t0
    logging.info(f"Generation took {elapsed:.1f}s ({elapsed/STEPS:.1f}s/step)")

    # Step 4: Save
    out_path = os.path.join(OUTPUT_DIR, f"test_lora_{test_name}.mp4")
    export_to_video(output.frames[0], out_path, fps=16)
    logging.info(f"Saved: {out_path}")

    del output
    gc.collect()
    torch.cuda.empty_cache()

    return out_path


def main():
    parser = argparse.ArgumentParser(description="I2V LoRA test script")
    parser.add_argument("--test", required=True,
                        choices=["none", "ulzzang", "uka", "both", "hipsway", "all"],
                        help="Test scenario to run")
    args = parser.parse_args()

    # Verify files exist
    for path, name in [(I2V_GGUF_HIGH, "GGUF HighNoise"), (I2V_GGUF_LOW, "GGUF LowNoise"),
                        (ULZZANG_PATH, "UlzzangG1"), (UKA_PATH, "UkaSexyLight"),
                        (HIPSWAY_PATH, "HipSway"), (IMAGE_PATH, "Reference image")]:
        if not os.path.exists(path):
            logging.error(f"Missing: {name} at {path}")
            sys.exit(1)
        logging.info(f"Found: {name} ({path})")

    # Load pipeline
    pipe = load_pipeline()

    # Debug PEFT config
    debug_peft_config(pipe)

    # Define test scenarios
    tests = {
        "none": {
            "name": "none_base_model",
            "adapters": None,
            "weights": None,
        },
        "ulzzang": {
            "name": "ulzzang_low085",
            "adapters": ["UlzzangG1_low"],
            "weights": [0.85],
        },
        "uka": {
            "name": "uka_high100",
            "adapters": ["UkaSexyLight_high"],
            "weights": [1.0],
        },
        "both": {
            "name": "both_ulzzang_uka",
            "adapters": ["UkaSexyLight_high", "UlzzangG1_low"],
            "weights": [1.0, 0.85],
        },
        "hipsway": {
            "name": "hipsway_high100",
            "adapters": ["HipSway_high", "UlzzangG1_low"],
            "weights": [1.0, 0.85],
            "prompt": "UlzzangG1, a beautiful young korean woman with large eyes, pale skin, makeup and small full lips, She sways her hips side to side with her arms crossed, high quality, smooth movement",
        },
    }

    if args.test == "all":
        for key in ["none", "ulzzang", "uka", "both", "hipsway"]:
            t = tests[key]
            run_test(pipe, t["name"], t["adapters"], t["weights"], t.get("prompt"))
    else:
        t = tests[args.test]
        run_test(pipe, t["name"], t["adapters"], t["weights"], t.get("prompt"))

    logging.info("\nAll tests complete!")


if __name__ == "__main__":
    main()
