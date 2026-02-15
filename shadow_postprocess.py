"""
Post-processing: Add projected floor shadow beneath the dancer.
Uses MediaPipe Selfie Segmentation for fast per-frame person masking,
then projects a perspective-warped shadow onto the floor.
"""
import os
import cv2
import numpy as np
import logging
import subprocess

_mp_selfie = None


def _get_segmentor():
    """Lazy-load MediaPipe Selfie Segmentation (singleton)."""
    global _mp_selfie
    if _mp_selfie is not None:
        return _mp_selfie

    import mediapipe as mp
    _mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    logging.info("MediaPipe Selfie Segmentation loaded (model=1, landscape)")
    return _mp_selfie


def _create_floor_shadow(mask: np.ndarray, frame_h: int, frame_w: int,
                         shadow_opacity: float = 0.45,
                         blur_size: int = 51,
                         squash_factor: float = 0.15,
                         y_offset_ratio: float = 0.02) -> np.ndarray:
    """
    Create a floor shadow from a person segmentation mask.

    The shadow is created by:
    1. Finding the person's bottom edge (feet position)
    2. Squashing the mask vertically (flatten onto floor plane)
    3. Positioning it at the feet level
    4. Applying Gaussian blur for soft shadow edges
    5. Masking out the person area (shadow only on floor)

    Args:
        mask: Binary person mask (H, W), values 0-1 float
        frame_h, frame_w: Frame dimensions
        shadow_opacity: Shadow darkness (0=invisible, 1=black)
        blur_size: Gaussian blur kernel size (must be odd)
        squash_factor: How much to squash vertically (0.1=very flat, 0.5=tall)
        y_offset_ratio: Small downward shift as ratio of frame height

    Returns:
        Shadow alpha map (H, W), float 0-1, where 1 = full shadow
    """
    # Find person bounding box
    rows = np.any(mask > 0.5, axis=1)
    cols = np.any(mask > 0.5, axis=0)
    if not rows.any() or not cols.any():
        return np.zeros((frame_h, frame_w), dtype=np.float32)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    person_h = y_max - y_min
    person_w = x_max - x_min
    person_cx = (x_min + x_max) // 2
    feet_y = y_max  # Bottom of person = feet

    if person_h < 10 or person_w < 10:
        return np.zeros((frame_h, frame_w), dtype=np.float32)

    # Create squashed shadow by resizing the mask
    shadow_h = max(5, int(person_h * squash_factor))
    shadow_w = int(person_w * 1.1)  # Slightly wider

    # Extract person region mask and resize to shadow dimensions
    person_region = mask[y_min:y_max, x_min:x_max]
    shadow_squashed = cv2.resize(person_region, (shadow_w, shadow_h),
                                  interpolation=cv2.INTER_AREA)

    # Position shadow at feet level
    shadow_canvas = np.zeros((frame_h, frame_w), dtype=np.float32)
    shadow_x_start = person_cx - shadow_w // 2
    shadow_y_start = feet_y + int(frame_h * y_offset_ratio)

    # Clip to frame boundaries
    src_x_start = max(0, -shadow_x_start)
    src_y_start = max(0, -shadow_y_start)
    dst_x_start = max(0, shadow_x_start)
    dst_y_start = max(0, shadow_y_start)
    dst_x_end = min(frame_w, shadow_x_start + shadow_w)
    dst_y_end = min(frame_h, shadow_y_start + shadow_h)
    src_x_end = src_x_start + (dst_x_end - dst_x_start)
    src_y_end = src_y_start + (dst_y_end - dst_y_start)

    if dst_x_end > dst_x_start and dst_y_end > dst_y_start:
        shadow_canvas[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            shadow_squashed[src_y_start:src_y_end, src_x_start:src_x_end]

    # Apply Gaussian blur for soft shadow
    if blur_size % 2 == 0:
        blur_size += 1
    shadow_canvas = cv2.GaussianBlur(shadow_canvas, (blur_size, blur_size), 0)

    # Remove shadow from person area (shadow only on floor, not on person)
    shadow_canvas = shadow_canvas * (1.0 - mask)

    # Apply opacity
    shadow_canvas = shadow_canvas * shadow_opacity

    return shadow_canvas


def add_shadow_to_video(input_path: str, shadow_opacity: float = 0.45) -> str:
    """
    Add projected floor shadow to a dance video.

    Processes each frame:
    1. MediaPipe segmentation → person mask
    2. Project shadow onto floor beneath person
    3. Composite shadow + original frame

    Args:
        input_path: Path to input video.
        shadow_opacity: Shadow intensity (0.0-1.0).

    Returns:
        Path to output video (replaces input in-place).
    """
    segmentor = _get_segmentor()

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        logging.warning(f"Shadow: Invalid video (fps={fps}, frames={total_frames})")
        return input_path

    logging.info(f"Shadow: Processing {total_frames} frames ({w}x{h}, {fps:.1f}fps)")

    output_path = input_path.replace(".mp4", "_shadow.mp4")
    if output_path == input_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_shadow{ext}"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = segmentor.process(rgb)

        if result.segmentation_mask is not None:
            mask = result.segmentation_mask  # float32, 0-1, (H, W)

            # Create floor shadow
            shadow_alpha = _create_floor_shadow(
                mask, h, w,
                shadow_opacity=shadow_opacity,
                blur_size=max(31, min(101, w // 15)),  # Scale blur with resolution
                squash_factor=0.15,
                y_offset_ratio=0.01,
            )

            # Composite: darken frame where shadow exists
            # shadow_alpha is 0-1, where 1 = full shadow (darkest)
            frame_float = frame.astype(np.float32)
            shadow_3ch = shadow_alpha[:, :, np.newaxis]
            frame_float = frame_float * (1.0 - shadow_3ch)
            frame = frame_float.clip(0, 255).astype(np.uint8)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            logging.info(f"Shadow: Processed {frame_idx}/{total_frames} frames")

    cap.release()
    writer.release()
    logging.info(f"Shadow: Processing complete ({frame_idx} frames)")

    # Re-encode to H.264
    h264_path = output_path.replace("_shadow.mp4", "_shadow_h264.mp4")
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", output_path,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-an",
            h264_path,
        ], capture_output=True, timeout=600)
        if result.returncode == 0 and os.path.exists(h264_path) and os.path.getsize(h264_path) > 0:
            os.replace(h264_path, input_path)
            os.remove(output_path)
            logging.info(f"Shadow: Saved to {input_path}")
        else:
            logging.warning(f"Shadow: H.264 re-encode failed, using mp4v")
            os.replace(output_path, input_path)
    except Exception as e:
        logging.warning(f"Shadow: H.264 re-encode error: {e}")
        os.replace(output_path, input_path)

    return input_path
