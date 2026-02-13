#!/usr/bin/env python3
"""Batch face swap test: directly via ComfyUI API."""

import requests
import glob
import json
import time
import os
import shutil
import uuid
import random
import logging

import cv2
import mediapipe as mp

logging.basicConfig(level=logging.INFO, format="%(message)s")

COMFYUI_URL = "http://127.0.0.1:8188"
WORKFLOW_PATH = "workflow/flux_klein_faceswap_api.json"
AVATAR_PATH = "settings/avatars/yuna/yuna_full_body.jpeg"
UPLOADS_DIR = "uploads"
OUTPUT_DIR = "outputs"
MODEL_PATH = "models/blaze_face_short_range.tflite"


def crop_face_head_standalone(image_path, padding_ratio=2.5):
    """Standalone crop_face_head — same logic as server.py."""
    img = cv2.imread(image_path)
    if img is None:
        return image_path

    h, w = img.shape[:2]
    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        min_detection_confidence=0.4,
    )

    detections = None
    for strategy in ["original", "upper_half", "upper_third"]:
        try:
            if strategy == "original":
                detect_img = img
            elif strategy == "upper_half":
                detect_img = img[:h // 2, :]
            else:
                detect_img = img[:h // 3, :]

            rgb = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
                result = detector.detect(mp_image)
            if result.detections:
                detections = result.detections
                logging.info(f"  Face found via '{strategy}'")
                break
        except Exception as e:
            logging.warning(f"  {strategy} failed: {e}")
            continue

    if not detections:
        logging.warning(f"  No face detected, returning as-is")
        return image_path

    det = detections[0]
    bb = det.bounding_box
    fx, fy, fw, fh = bb.origin_x, bb.origin_y, bb.width, bb.height

    face_area_ratio = (fw * fh) / (w * h)
    if face_area_ratio > 0.15:
        logging.info(f"  Face is {face_area_ratio:.1%} of image, skipping crop")
        return image_path

    cx, cy = fx + fw // 2, fy + fh // 2
    half_size = int(max(fw, fh) * padding_ratio / 2)
    x1 = max(0, cx - half_size)
    y1 = max(0, cy - int(half_size * 1.0))
    x2 = min(w, cx + half_size)
    y2 = min(h, cy + int(half_size * 1.5))

    cropped = img[y1:y2, x1:x2]
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=OUTPUT_DIR)
    cv2.imwrite(tmp.name, cropped)
    tmp.close()
    logging.info(f"  Portrait crop: ({x2-x1}x{y2-y1}), face area={face_area_ratio:.1%}")
    return tmp.name


def upload_to_comfyui(filepath):
    fname = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        resp = requests.post(
            f"{COMFYUI_URL}/upload/image",
            files={"image": (fname, f, "image/png")},
            data={"overwrite": "true"},
        )
    resp.raise_for_status()
    return resp.json()["name"]


def run_faceswap(avatar_comfyui, style_comfyui, workflow_template):
    wf = json.loads(json.dumps(workflow_template))
    wf["10"]["inputs"]["image"] = style_comfyui
    wf["11"]["inputs"]["image"] = avatar_comfyui
    wf["92"]["inputs"]["value"] = random.randint(0, 2**53)

    client_id = str(uuid.uuid4())
    resp = requests.post(
        f"{COMFYUI_URL}/prompt",
        json={"prompt": wf, "client_id": client_id},
    )
    if resp.status_code != 200:
        return None, f"Rejected: {resp.text[:200]}"

    prompt_id = resp.json()["prompt_id"]
    for _ in range(120):
        time.sleep(2)
        try:
            hist = requests.get(f"{COMFYUI_URL}/history/{prompt_id}").json()
            if prompt_id in hist:
                status = hist[prompt_id].get("status", {})
                if status.get("status_str") == "error":
                    return None, "ComfyUI error"
                outputs = hist[prompt_id].get("outputs", {})
                for node_out in outputs.values():
                    for img in node_out.get("images", []):
                        return img["filename"], None
        except Exception:
            pass
    return None, "Timeout"


def main():
    with open(WORKFLOW_PATH) as f:
        workflow_template = json.load(f)

    images = sorted(
        glob.glob(f"{UPLOADS_DIR}/*.jpg")
        + glob.glob(f"{UPLOADS_DIR}/*.jpeg")
        + glob.glob(f"{UPLOADS_DIR}/*.png")
    )
    print(f"Found {len(images)} images")
    print(f"Avatar: {AVATAR_PATH}")
    print(f"{'='*60}")

    # Crop avatar
    print(f"\nPreparing avatar...")
    cropped = crop_face_head_standalone(AVATAR_PATH)
    avatar_comfyui = upload_to_comfyui(cropped)
    print(f"  ComfyUI: {avatar_comfyui}")
    if cropped != AVATAR_PATH and os.path.exists(cropped):
        os.remove(cropped)

    results = []
    for i, img_path in enumerate(images):
        fname = os.path.basename(img_path)
        print(f"\n[{i+1}/{len(images)}] {fname}")

        style_comfyui = upload_to_comfyui(img_path)
        t0 = time.time()
        output_file, error = run_faceswap(avatar_comfyui, style_comfyui, workflow_template)
        elapsed = time.time() - t0

        if error:
            print(f"  FAILED ({elapsed:.1f}s): {error}")
            results.append((fname, "FAILED", error))
        else:
            src = f"/home/ubuntu/ComfyUI/output/{output_file}"
            dst = f"{OUTPUT_DIR}/test_faceswap_{i+1:02d}.png"
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"  OK ({elapsed:.1f}s) -> {dst}")
            else:
                dst = output_file
                print(f"  OK ({elapsed:.1f}s) -> {dst} (ComfyUI)")
            results.append((fname, "OK", dst))

    print(f"\n{'='*60}")
    print(f"SUMMARY: {sum(1 for _,s,_ in results if s=='OK')}/{len(results)} success")
    for fname, status, out in results:
        print(f"  [{status}] {fname} -> {out}")


if __name__ == "__main__":
    main()
