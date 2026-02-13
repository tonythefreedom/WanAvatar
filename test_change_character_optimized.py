#!/usr/bin/env python3
"""
Test script for Change Character V1.1 with all optimizations:
- TeaCache (30-45% speedup)
- torch.compile reduce-overhead (20-40% speedup)
- batched_cfg (15-25% speedup)
- FP8 quantization (10-15% speedup)
- rms_norm pytorch + merge_loras

Usage: python test_change_character_optimized.py
"""

import requests
import time
import sys
import json
import os
import datetime

# Generate JWT token for auth
import jwt as pyjwt

JWT_SECRET = os.environ.get("JWT_SECRET", "def60bb49cec3fafcfebeaac02a6fec23c83dd34b86cb252bcb25379b8a8c074")

def make_token():
    return pyjwt.encode(
        {"sub": "1", "email": "tony@banya.ai", "role": "superadmin",
         "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)},
        JWT_SECRET, algorithm="HS256",
    )

SERVER_URL = "http://localhost:8000"
AUTH_HEADER = {"Authorization": f"Bearer {make_token()}"}

AVATAR = "outputs/upload_8b439dac-44cd-4530-94d6-feeb1c79b1d4.jpg"
VIDEO = "uploads/ff9d32a6-5ba0-42c9-9c04-e675010a52d8.mp4"
BG_IMAGE = "background/stages/dance_stg_fanta.png"
PROMPT = "The character is dancing energetically on a fantasy stage with vibrant lighting"


def main():
    print("=" * 60)
    print("Change Character V1.1 - Optimized Speed Test")
    print("=" * 60)
    print(f"Avatar:     {AVATAR}")
    print(f"Video:      {VIDEO}")
    print(f"Background: {BG_IMAGE}")
    print(f"Prompt:     {PROMPT}")
    print()

    # Step 1: Start generation via API
    print("[1/3] Starting generation...")
    payload = {
        "workflow_id": "change_character",
        "inputs": {
            "ref_image": AVATAR,
            "ref_video": VIDEO,
            "prompt": PROMPT,
            "aspect_ratio": "portrait",
            "bg_image": BG_IMAGE,
        },
    }

    try:
        resp = requests.post(f"{SERVER_URL}/api/workflow/generate", json=payload, headers=AUTH_HEADER, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        task_id = data["task_id"]
        print(f"  Task ID: {task_id}")
    except Exception as e:
        print(f"  FAILED to start: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response: {e.response.text[:500]}")
        sys.exit(1)

    # Step 2: Monitor progress
    print("\n[2/3] Monitoring progress...")
    start_time = time.time()
    last_progress = -1
    last_message = ""

    while True:
        try:
            resp = requests.get(f"{SERVER_URL}/api/status/{task_id}", headers=AUTH_HEADER, timeout=10)
            status = resp.json()

            progress = round((status.get("progress", 0) or 0) * 100)
            message = status.get("message", "")
            task_status = status.get("status", "unknown")
            elapsed = time.time() - start_time

            if progress != last_progress or message != last_message:
                mins, secs = divmod(int(elapsed), 60)
                print(f"  [{mins:02d}:{secs:02d}] {progress:3d}% | {message}")
                last_progress = progress
                last_message = message

            if task_status == "completed":
                elapsed = time.time() - start_time
                mins, secs = divmod(int(elapsed), 60)
                print(f"\n  COMPLETED in {mins}m {secs}s")
                output_url = status.get("output_url") or status.get("output_path", "")
                print(f"  Output: {output_url}")
                break
            elif task_status == "failed":
                print(f"\n  FAILED: {status.get('message', 'Unknown error')}")
                sys.exit(1)
            elif task_status == "cancelled":
                print(f"\n  CANCELLED")
                sys.exit(1)

        except Exception as e:
            print(f"  Poll error: {e}")

        time.sleep(5)

    # Step 3: Summary
    total_time = time.time() - start_time
    mins, secs = divmod(int(total_time), 60)
    print(f"\n[3/3] Summary")
    print("=" * 60)
    print(f"Total time:    {mins}m {secs}s")
    print(f"Optimizations: TeaCache + torch.compile + batched_cfg + FP8 + rms_norm_pytorch + merge_loras")
    print(f"Resolution:    1080x1920 (Portrait)")
    print(f"Video length:  ~37s source")
    print("=" * 60)


if __name__ == "__main__":
    main()
