#!/usr/bin/env python3
"""
Preprocessing script for LoRA fine-tuning data.
Extracts frames, audio, and generates face/lip masks from video files.
Uses MediaPipe Tasks API (0.10.x).
"""

import os
import cv2
import numpy as np
from pathlib import Path
import subprocess
import argparse
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# MediaPipe Face Mesh landmarks for lips
LIP_LANDMARKS = [
    # Outer lips
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    # Inner lips
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    # Upper lip
    185, 40, 39, 37, 0, 267, 269, 270, 409,
    # Lower lip
    191, 80, 81, 82, 13, 312, 311, 310, 415
]

# Full face landmarks (convex hull)
FACE_OVAL_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]


def extract_frames(video_path: str, output_dir: str, fps: int = 25) -> int:
    """Extract frames from video at specified fps."""
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f'fps={fps}',
        '-q:v', '2',
        os.path.join(output_dir, '%04d.png'),
        '-y', '-loglevel', 'error'
    ]

    subprocess.run(cmd, check=True)

    # Count extracted frames
    return len([f for f in os.listdir(output_dir) if f.endswith('.png')])


def extract_audio(video_path: str, output_path: str, sample_rate: int = 16000) -> None:
    """Extract audio from video at specified sample rate."""
    cmd = [
        'ffmpeg', '-i', video_path,
        '-ar', str(sample_rate),
        '-ac', '1',
        '-vn',
        output_path,
        '-y', '-loglevel', 'error'
    ]

    subprocess.run(cmd, check=True)


def create_mask_from_landmarks(image_shape: tuple, landmarks: list, indices: list) -> np.ndarray:
    """Create binary mask from face mesh landmarks."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    points = []
    for idx in indices:
        if idx < len(landmarks):
            lm = landmarks[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append([x, y])

    if len(points) > 2:
        points = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)

    return mask


class FaceMaskGenerator:
    """Face and lip mask generator using MediaPipe Tasks API."""

    def __init__(self, model_path: str):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def generate_masks(self, image: np.ndarray) -> tuple:
        """Generate face and lip masks for an image."""
        h, w = image.shape[:2]
        face_mask = np.zeros((h, w), dtype=np.uint8)
        lip_mask = np.zeros((h, w), dtype=np.uint8)

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Detect face landmarks
        results = self.detector.detect(mp_image)

        if results.face_landmarks and len(results.face_landmarks) > 0:
            landmarks = results.face_landmarks[0]

            # Generate face mask
            face_mask = create_mask_from_landmarks(image.shape, landmarks, FACE_OVAL_LANDMARKS)

            # Generate lip mask
            lip_mask = create_mask_from_landmarks(image.shape, landmarks, LIP_LANDMARKS)

            # Dilate lip mask slightly for better coverage
            kernel = np.ones((5, 5), np.uint8)
            lip_mask = cv2.dilate(lip_mask, kernel, iterations=1)

        return face_mask, lip_mask

    def close(self):
        self.detector.close()


def process_masks_for_directory(images_dir: str, face_masks_dir: str, lip_masks_dir: str, model_path: str) -> None:
    """Generate face and lip masks for all images in directory."""
    os.makedirs(face_masks_dir, exist_ok=True)
    os.makedirs(lip_masks_dir, exist_ok=True)

    generator = FaceMaskGenerator(model_path)

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Warning: Could not read {img_path}")
            continue

        face_mask, lip_mask = generator.generate_masks(image)

        # Save masks
        cv2.imwrite(os.path.join(face_masks_dir, img_file), face_mask)
        cv2.imwrite(os.path.join(lip_masks_dir, img_file), lip_mask)

    generator.close()


def copy_original_video(src_path: str, dst_path: str) -> None:
    """Copy original video as sub_clip.mp4."""
    subprocess.run(['cp', src_path, dst_path], check=True)


def process_video(video_path: str, output_base_dir: str, video_idx: int, model_path: str, fps: int = 25) -> str:
    """Process a single video file."""
    video_dir = os.path.join(output_base_dir, f'video_{video_idx:03d}')
    os.makedirs(video_dir, exist_ok=True)

    images_dir = os.path.join(video_dir, 'images')
    face_masks_dir = os.path.join(video_dir, 'face_masks')
    lip_masks_dir = os.path.join(video_dir, 'lip_masks')

    # 1. Copy original video
    sub_clip_path = os.path.join(video_dir, 'sub_clip.mp4')
    copy_original_video(video_path, sub_clip_path)

    # 2. Extract frames
    num_frames = extract_frames(video_path, images_dir, fps)

    # 3. Extract audio
    audio_path = os.path.join(video_dir, 'audio.wav')
    extract_audio(video_path, audio_path)

    # 4. Generate masks
    process_masks_for_directory(images_dir, face_masks_dir, lip_masks_dir, model_path)

    return video_dir


def main():
    parser = argparse.ArgumentParser(description='Preprocess videos for LoRA training')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing mp4 files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed data')
    parser.add_argument('--fps', type=int, default=25, help='Target frame rate (default: 25)')
    parser.add_argument('--model_path', type=str, default='face_landmarker.task',
                        help='Path to MediaPipe face landmarker model')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    # Resolve model path
    model_path = args.model_path
    if not os.path.isabs(model_path):
        # Try to find it relative to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, model_path)

    if not os.path.exists(model_path):
        print(f"Error: MediaPipe model not found at {model_path}")
        print("Please download it from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Find all mp4 files
    video_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.mp4')])

    print(f"Found {len(video_files)} video files")
    print(f"Using MediaPipe model: {model_path}")

    processed_dirs = []

    for idx, video_file in enumerate(tqdm(video_files, desc="Processing videos"), start=1):
        video_path = os.path.join(input_dir, video_file)
        print(f"\nProcessing [{idx}/{len(video_files)}]: {video_file}")

        try:
            video_dir = process_video(video_path, output_dir, idx, model_path, args.fps)
            processed_dirs.append(video_dir)
            print(f"  -> Saved to {video_dir}")
        except Exception as e:
            print(f"  -> Error: {e}")
            import traceback
            traceback.print_exc()

    # Create video_path.txt
    path_file = os.path.join(output_dir, 'video_path.txt')
    with open(path_file, 'w') as f:
        for d in processed_dirs:
            f.write(os.path.abspath(d) + '\n')

    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Total videos processed: {len(processed_dirs)}")
    print(f"Video path file: {path_file}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
