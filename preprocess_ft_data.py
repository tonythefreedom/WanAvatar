#!/usr/bin/env python3
"""
Preprocess FT_data videos for LoRA fine-tuning.

For each video in FT_data/:
1. Create structured directory
2. Copy video as sub_clip.mp4
3. Extract audio as audio.wav (16kHz)
4. Extract frames as images/frame_XXX.png
5. Generate face masks using MediaPipe FaceLandmarker
6. Generate lip masks using MediaPipe FaceLandmarker
"""

import os
import sys
import cv2
import shutil
import subprocess
import numpy as np
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions


# MediaPipe Face Mesh landmark indices (478 landmarks total)
# Full face contour (silhouette)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Outer lip contour
LIP_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
             409, 270, 269, 267, 0, 37, 39, 40, 185]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker_v2_with_blendshapes.task")


def extract_audio(video_path, output_path, sr=16000):
    """Extract audio from video at specified sample rate."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sr), "-ac", "1",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    return result.returncode == 0


def extract_frames(video_path, output_dir):
    """Extract all frames from video as PNG files."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_idx += 1

    cap.release()
    return frame_idx, fps


def create_mask_from_landmarks(landmarks, indices, img_w, img_h):
    """Create a binary mask from face landmark coordinates."""
    points = []
    for idx in indices:
        lm = landmarks[idx]
        x = int(lm.x * img_w)
        y = int(lm.y * img_h)
        points.append([x, y])

    points = np.array(points, dtype=np.int32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, points, 255)
    return mask


def create_lip_mask_from_landmarks(landmarks, img_w, img_h):
    """Create lip mask from outer lip contour."""
    outer_points = []
    for idx in LIP_OUTER:
        lm = landmarks[idx]
        outer_points.append([int(lm.x * img_w), int(lm.y * img_h)])
    outer_points = np.array(outer_points, dtype=np.int32)

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [outer_points], 255)
    return mask


def generate_masks(frames_dir, face_masks_dir, lip_masks_dir):
    """Generate face and lip masks for all frames using MediaPipe FaceLandmarker."""
    os.makedirs(face_masks_dir, exist_ok=True)
    os.makedirs(lip_masks_dir, exist_ok=True)

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    last_face_mask = None
    last_lip_mask = None

    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        img = cv2.imread(frame_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # Convert to MediaPipe Image
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = landmarker.detect(mp_image)

        if results.face_landmarks and len(results.face_landmarks) > 0:
            landmarks = results.face_landmarks[0]

            # Face mask (oval contour)
            face_mask = create_mask_from_landmarks(landmarks, FACE_OVAL, img_w, img_h)
            # Lip mask
            lip_mask = create_lip_mask_from_landmarks(landmarks, img_w, img_h)

            last_face_mask = face_mask
            last_lip_mask = lip_mask
        else:
            # Use last known mask or white mask
            if last_face_mask is not None:
                face_mask = last_face_mask
                lip_mask = last_lip_mask
            else:
                face_mask = np.ones((img_h, img_w), dtype=np.uint8) * 255
                lip_mask = np.ones((img_h, img_w), dtype=np.uint8) * 255

        cv2.imwrite(os.path.join(face_masks_dir, frame_file), face_mask)
        cv2.imwrite(os.path.join(lip_masks_dir, frame_file), lip_mask)

    landmarker.close()
    return len(frame_files)


def process_video(video_path, output_base_dir, video_idx):
    """Process a single video into training format."""
    video_name = f"video_{video_idx:03d}"
    video_dir = os.path.join(output_base_dir, video_name)

    if os.path.exists(video_dir):
        # Check if already fully processed
        required = ["sub_clip.mp4", "audio.wav", "images", "face_masks", "lip_masks"]
        if all(os.path.exists(os.path.join(video_dir, r)) for r in required):
            images_dir = os.path.join(video_dir, "images")
            n_frames = len([f for f in os.listdir(images_dir) if f.endswith('.png')])
            if n_frames > 0:
                print(f"  [{video_idx}] Already processed: {video_name} ({n_frames} frames)")
                return video_dir

    os.makedirs(video_dir, exist_ok=True)

    # 1. Copy video
    dest_video = os.path.join(video_dir, "sub_clip.mp4")
    shutil.copy2(video_path, dest_video)

    # 2. Extract audio at 16kHz
    audio_path = os.path.join(video_dir, "audio.wav")
    if not extract_audio(video_path, audio_path):
        print(f"  [{video_idx}] WARNING: Failed to extract audio from {video_path}")
        return None

    # 3. Extract frames
    images_dir = os.path.join(video_dir, "images")
    n_frames, fps = extract_frames(dest_video, images_dir)
    if n_frames == 0:
        print(f"  [{video_idx}] WARNING: No frames extracted from {video_path}")
        return None

    # 4. Generate face/lip masks
    face_masks_dir = os.path.join(video_dir, "face_masks")
    lip_masks_dir = os.path.join(video_dir, "lip_masks")
    n_masks = generate_masks(images_dir, face_masks_dir, lip_masks_dir)

    print(f"  [{video_idx}] {video_name}: {n_frames} frames, {fps:.1f} fps, {n_masks} masks")
    return video_dir


def main():
    ft_data_dir = "/home/ubuntu/WanAvatar/FT_data"
    output_base_dir = "/home/ubuntu/WanAvatar/FT_data/processed"
    os.makedirs(output_base_dir, exist_ok=True)

    # Find all MP4 files
    video_files = sorted([
        os.path.join(ft_data_dir, f)
        for f in os.listdir(ft_data_dir)
        if f.endswith('.mp4')
    ])

    print(f"Found {len(video_files)} videos in {ft_data_dir}")
    print(f"Output directory: {output_base_dir}")
    print()

    video_dirs = []
    for idx, video_path in enumerate(video_files):
        print(f"Processing [{idx+1}/{len(video_files)}]: {os.path.basename(video_path)}")
        result = process_video(video_path, output_base_dir, idx)
        if result:
            video_dirs.append(result)

    # Write video_path.txt
    txt_path = os.path.join(output_base_dir, "video_path.txt")
    with open(txt_path, 'w') as f:
        for d in video_dirs:
            f.write(d + '\n')

    print(f"\nDone! Processed {len(video_dirs)}/{len(video_files)} videos")
    print(f"Video path list: {txt_path}")

    # Also create validation data from the reference image + first video's audio
    validation_dir = "/home/ubuntu/WanAvatar/validation"
    os.makedirs(validation_dir, exist_ok=True)

    ref_image = os.path.join(ft_data_dir, "reference.jpeg")
    if os.path.exists(ref_image):
        dest_ref = os.path.join(validation_dir, "reference.png")
        from PIL import Image
        img = Image.open(ref_image).convert('RGB')
        img.save(dest_ref)
        print(f"Validation reference: {dest_ref}")

    # Use first video's audio for validation
    if video_dirs:
        first_audio = os.path.join(video_dirs[0], "audio.wav")
        dest_audio = os.path.join(validation_dir, "audio.wav")
        shutil.copy2(first_audio, dest_audio)
        print(f"Validation audio: {dest_audio}")


if __name__ == "__main__":
    main()
