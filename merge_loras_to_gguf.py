#!/usr/bin/env python3
"""
Merge multiple LoRAs into WanVideo bf16 base model and save as safetensors.
Then the merged model can be converted to GGUF and quantized.

Handles multiple LoRA formats:
- Standard: lora_A/lora_B + alpha
- Kohya: lora_unet__*_lora_down/up + alpha
- diffusion_model: diffusion_model.*.lora_down/up
- diff_b: bias diffs
- Non-LoRA modules (face_adapter etc.) - added directly to state dict

Usage:
    python merge_loras_to_gguf.py
"""

import torch
import re
import sys
import os
import gc
from pathlib import Path
from safetensors.torch import load_file, save_file

# ============================================================
# Configuration
# ============================================================
BASE_MODEL = "/home/ubuntu/ComfyUI/models/diffusion_models/Wan22Animate/wan2.2_animate_14B_bf16.safetensors"
OUTPUT_MERGED = "/mnt/models/Wan22Animate_merged_loras_bf16.safetensors"

LORAS_DIR = "/home/ubuntu/ComfyUI/models/loras"

# LoRAs to merge (same order and strengths as workflow)
LORAS = [
    {"file": "lightx2v_elite_it2v_animate_face.safetensors", "strength": 1.2},
    {"file": "WAN22_MoCap_fullbodyCOPY_ED.safetensors",      "strength": 0.4, "type": "module"},
    {"file": "Wan2.2-Fun-A14B-InP-Fusion-Elite.safetensors",  "strength": 0.65},
    {"file": "FullDynamic_Ultimate_Fusion_Elite.safetensors",  "strength": 0.7},
    {"file": "WanAnimate_relight_lora_fp16.safetensors",       "strength": 0.7},
]


def build_kohya_key_map(base_keys):
    """Build a mapping from Kohya underscore-encoded keys to actual model keys.
    Kohya encodes 'blocks.0.cross_attn.k' as 'blocks_0_cross_attn_k'.
    We build a reverse map from the base model's actual keys.
    """
    key_map = {}
    for base_key in base_keys:
        # Map with .weight suffix stripped (LoRA keys don't include .weight in prefix)
        stripped = base_key
        if stripped.endswith(".weight"):
            stripped = stripped[:-len(".weight")]
        elif stripped.endswith(".bias"):
            stripped = stripped[:-len(".bias")]
        kohya_key = stripped.replace(".", "_")
        key_map[kohya_key] = stripped
    return key_map

_kohya_map = None

def standardize_lora_keys(lora_sd, base_keys=None):
    """Convert various LoRA key formats to standard format."""
    global _kohya_map
    if base_keys is not None and _kohya_map is None:
        _kohya_map = build_kohya_key_map(base_keys)

    new_sd = {}
    for key, val in lora_sd.items():
        new_key = key

        # Handle Kohya format: lora_unet__blocks_0_cross_attn_k.alpha
        if new_key.startswith("lora_unet__"):
            new_key = new_key[len("lora_unet__"):]
            # Split off the LoRA suffix (.alpha, .lora_down.weight, etc.)
            suffix = ""
            for s in [".alpha", ".lora_down.weight", ".lora_up.weight", ".diff_b", ".diff"]:
                if new_key.endswith(s):
                    suffix = s
                    new_key = new_key[:len(new_key) - len(s)]
                    break

            # Use the kohya map to find the real key
            if _kohya_map and new_key in _kohya_map:
                new_key = _kohya_map[new_key]
            else:
                # Fallback: replace underscores with dots
                new_key = new_key.replace("_", ".")

            new_key = new_key + suffix

        elif new_key.startswith("lora_unet_"):
            new_key = new_key[len("lora_unet_"):]
            new_key = new_key.replace("__", ".")

        # Ensure diffusion_model prefix for block keys
        if not new_key.startswith("diffusion_model."):
            if any(new_key.startswith(p) for p in ["blocks.", "head.", "text_embedding.", "patch_embedding."]):
                new_key = "diffusion_model." + new_key

        # Standardize LoRA weight names
        new_key = new_key.replace(".lora_down.weight", ".lora_A.weight")
        new_key = new_key.replace(".lora_up.weight", ".lora_B.weight")

        new_sd[new_key] = val
    return new_sd


def is_lora_file(lora_sd):
    """Check if the state dict contains LoRA weights (vs full module weights)."""
    return any("lora" in k.lower() for k in lora_sd.keys())


def find_lora_pairs(lora_sd):
    """Find all LoRA A/B pairs, alpha values, and bias diffs."""
    pairs = {}
    bias_diffs = {}

    for key in lora_sd:
        if ".lora_A.weight" in key:
            base = key.replace(".lora_A.weight", "")
            if base not in pairs:
                pairs[base] = {}
            pairs[base]["A"] = lora_sd[key]

        elif ".lora_B.weight" in key:
            base = key.replace(".lora_B.weight", "")
            if base not in pairs:
                pairs[base] = {}
            pairs[base]["B"] = lora_sd[key]

        elif key.endswith(".alpha"):
            base = key[:-len(".alpha")]
            if base not in pairs:
                pairs[base] = {}
            pairs[base]["alpha"] = lora_sd[key]

        elif ".diff_b" in key or key.endswith(".diff"):
            bias_diffs[key] = lora_sd[key]

    return pairs, bias_diffs


def get_base_key(lora_base_key):
    """Extract the base model key from a LoRA base key.
    e.g. 'diffusion_model.blocks.0.self_attn.q' -> 'blocks.0.self_attn.q.weight'
    """
    key = lora_base_key
    if key.startswith("diffusion_model."):
        key = key[len("diffusion_model."):]
    return key + ".weight"


def merge_lora(base_sd, lora_sd, strength, lora_name):
    """Merge a standard LoRA into the base state dict."""
    lora_sd = standardize_lora_keys(lora_sd, base_keys=list(base_sd.keys()))
    pairs, bias_diffs = find_lora_pairs(lora_sd)

    merged_count = 0
    skipped_count = 0
    bias_count = 0
    last_scale = 0.0

    for lora_base_key, components in pairs.items():
        if "A" not in components or "B" not in components:
            skipped_count += 1
            continue

        lora_A = components["A"]
        lora_B = components["B"]
        alpha = components.get("alpha", None)

        if alpha is not None:
            alpha_val = alpha.item() if alpha.numel() == 1 else alpha.float().mean().item()
        else:
            alpha_val = float(lora_A.shape[0])  # Default alpha = rank

        rank = lora_A.shape[0]
        last_scale = (alpha_val / rank) * strength

        # Find matching base model key
        base_key = get_base_key(lora_base_key)

        if base_key not in base_sd:
            # Try without .weight
            alt_key = base_key.replace(".weight", "")
            if alt_key in base_sd:
                base_key = alt_key
            else:
                skipped_count += 1
                continue

        # Compute LoRA diff: B @ A
        orig_shape = base_sd[base_key].shape
        try:
            diff = torch.mm(
                lora_B.float().flatten(start_dim=1),
                lora_A.float().flatten(start_dim=1)
            ).reshape(orig_shape)
        except RuntimeError as e:
            print(f"    Shape mismatch for {base_key}: base={orig_shape}, "
                  f"B={lora_B.shape}, A={lora_A.shape} - {e}")
            skipped_count += 1
            continue

        # Merge
        base_sd[base_key] = (base_sd[base_key].float() + last_scale * diff).to(base_sd[base_key].dtype)
        merged_count += 1

    # Handle bias diffs
    for diff_key, diff_val in bias_diffs.items():
        # diff_b key -> bias key
        base_key = diff_key.replace("diffusion_model.", "").replace(".diff_b", ".bias").replace(".diff", ".weight")
        if base_key in base_sd:
            base_sd[base_key] = (base_sd[base_key].float() + strength * diff_val.float()).to(base_sd[base_key].dtype)
            bias_count += 1

    scale_str = f", scale={last_scale:.4f}" if last_scale else ""
    print(f"  [{lora_name}] Merged {merged_count} weight layers + {bias_count} bias diffs, "
          f"skipped {skipped_count} (strength={strength}{scale_str})")
    return base_sd


def merge_module(base_sd, module_sd, strength, module_name):
    """Merge a full module (non-LoRA) into the base state dict.
    These are additional modules like face_adapter that get added to the model.
    For GGUF conversion, we add them directly to the state dict.
    """
    added = 0
    for key, val in module_sd.items():
        # These are standalone modules - add directly
        base_sd[key] = val
        added += 1

    print(f"  [{module_name}] Added {added} module tensors directly (face_adapter etc.)")
    return base_sd


def main():
    print("=" * 60)
    print("LoRA Merge Tool for WanVideo 14B")
    print("=" * 60)

    # Step 1: Load base model
    print(f"\n[1/3] Loading base model: {Path(BASE_MODEL).name}")
    print(f"  Size: {os.path.getsize(BASE_MODEL) / 1e9:.1f} GB")
    base_sd = load_file(BASE_MODEL)
    print(f"  Loaded {len(base_sd)} tensors")

    # Step 2: Merge each LoRA
    print(f"\n[2/3] Merging {len(LORAS)} LoRAs...")
    for i, lora_info in enumerate(LORAS):
        lora_path = os.path.join(LORAS_DIR, lora_info["file"])
        lora_type = lora_info.get("type", "lora")
        print(f"\n  Loading {i+1}/{len(LORAS)}: {lora_info['file']} (type={lora_type})")
        print(f"  Size: {os.path.getsize(lora_path) / 1e6:.0f} MB, Strength: {lora_info['strength']}")

        lora_sd = load_file(lora_path)

        if lora_type == "module" or not is_lora_file(lora_sd):
            base_sd = merge_module(base_sd, lora_sd, lora_info["strength"], lora_info["file"])
        else:
            base_sd = merge_lora(base_sd, lora_sd, lora_info["strength"], lora_info["file"])

        del lora_sd
        gc.collect()

    # Step 3: Save merged model
    print(f"\n[3/3] Saving merged model to: {OUTPUT_MERGED}")
    print(f"  Total tensors: {len(base_sd)}")
    save_file(base_sd, OUTPUT_MERGED)
    print(f"  Size: {os.path.getsize(OUTPUT_MERGED) / 1e9:.1f} GB")
    print(f"\n  Done! Next steps:")
    print(f"  1. Convert to GGUF: python convert.py --src {OUTPUT_MERGED}")
    print(f"  2. Quantize: llama-quantize <bf16.gguf> <q4_k_m.gguf> Q4_K_M")

    del base_sd
    gc.collect()


if __name__ == "__main__":
    main()
