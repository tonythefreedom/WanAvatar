"""
Post-processing: Add projected floor shadow beneath the dancer.
Uses MediaPipe Selfie Segmentation for fast per-frame person masking,
then projects a perspective-warped shadow onto the floor.
Shadow direction and color adapt to the background lighting.
"""
import os
import cv2
import numpy as np
import logging
import subprocess

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "models", "selfie_segmenter.tflite")


def _create_segmentor():
    """Create a NEW MediaPipe Selfie Segmenter instance.

    Each call creates a fresh instance to avoid thread-safety issues.
    MediaPipe VIDEO mode maintains internal timestamp state, so sharing
    a singleton between threads causes non-monotonic timestamp errors.
    """
    from mediapipe.tasks.python import vision, BaseOptions
    opts = vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        output_category_mask=False,
        output_confidence_masks=True,
        running_mode=vision.RunningMode.VIDEO,
    )
    segmentor = vision.ImageSegmenter.create_from_options(opts)
    logging.info("MediaPipe Selfie Segmenter created (Tasks API, VIDEO mode)")
    return segmentor


def _analyze_light_source(frame: np.ndarray, mask: np.ndarray) -> tuple:
    """Analyze background lighting to determine shadow direction and tint color.

    Examines the non-person areas of the frame to find the brightest region
    (light source) and the dominant floor color.

    Args:
        frame: BGR frame (H, W, 3), uint8
        mask: Person mask (H, W), float 0-1

    Returns:
        (light_offset_x, light_offset_y, shadow_color_bgr)
        - light_offset_x: horizontal shadow offset ratio (-1.0 to 1.0)
          negative = shadow goes left, positive = shadow goes right
        - light_offset_y: vertical offset (usually small, near 0)
        - shadow_color_bgr: (B, G, R) tuple for shadow tinting (0-255)
    """
    h, w = frame.shape[:2]
    # Create background-only frame (exclude person)
    bg_mask = (1.0 - mask)
    bg_mask_u8 = (bg_mask * 255).astype(np.uint8)

    # Convert to grayscale for brightness analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_bg = gray * bg_mask  # Only background pixels

    # Divide frame into a 3x3 grid and find brightest region
    grid_brightness = np.zeros((3, 3), dtype=np.float32)
    cell_h, cell_w = h // 3, w // 3
    for gy in range(3):
        for gx in range(3):
            y1, y2 = gy * cell_h, (gy + 1) * cell_h
            x1, x2 = gx * cell_w, (gx + 1) * cell_w
            cell = gray_bg[y1:y2, x1:x2]
            cell_mask = bg_mask[y1:y2, x1:x2]
            valid = cell_mask > 0.5
            if valid.any():
                grid_brightness[gy, gx] = np.mean(cell[valid])

    # Find brightest grid cell = light source position
    bright_idx = np.unravel_index(np.argmax(grid_brightness), grid_brightness.shape)
    light_gy, light_gx = bright_idx

    # Shadow goes opposite to light source
    # Light at left (gx=0) → shadow goes right (positive offset)
    # Light at top (gy=0) → shadow goes down (positive offset, but we keep small)
    light_offset_x = (1 - light_gx) * 0.15  # ±0.15 max horizontal offset
    light_offset_y = 0.0  # Keep vertical minimal

    # Sample floor color (bottom 25% of frame, excluding person)
    floor_region = frame[int(h * 0.75):, :]
    floor_mask_region = bg_mask[int(h * 0.75):, :]
    valid_floor = floor_mask_region > 0.5
    if valid_floor.any():
        floor_color = np.mean(frame[int(h * 0.75):][valid_floor], axis=0)
        # Darken the floor color for shadow tinting (30% brightness)
        shadow_color = (floor_color * 0.3).astype(np.uint8)
    else:
        shadow_color = np.array([20, 20, 20], dtype=np.uint8)  # Default dark

    return light_offset_x, light_offset_y, shadow_color


def _create_floor_shadow(mask: np.ndarray, frame_h: int, frame_w: int,
                         shadow_opacity: float = 0.55,
                         blur_size: int = 51,
                         squash_factor: float = 0.22,
                         y_offset_ratio: float = 0.02,
                         light_offset_x: float = 0.0,
                         shadow_color_bgr: tuple = None) -> tuple:
    """
    Create a floor shadow from a person segmentation mask.

    Two-layer approach:
    1. Contact shadow: sharp, dark, close to feet (ground contact realism)
    2. Ambient shadow: diffused, wider, lighter (overall shadow presence)

    Args:
        mask: Binary person mask (H, W), values 0-1 float
        frame_h, frame_w: Frame dimensions
        shadow_opacity: Shadow darkness (0=invisible, 1=black)
        blur_size: Gaussian blur kernel size (must be odd)
        squash_factor: How much to squash vertically (0.1=very flat, 0.5=tall)
        y_offset_ratio: Small downward shift as ratio of frame height
        light_offset_x: Horizontal offset for shadow direction (-1 to 1)
        shadow_color_bgr: (B, G, R) shadow tint color, or None for black

    Returns:
        (shadow_alpha, shadow_color_map)
        - shadow_alpha: (H, W) float 0-1, where 1 = full shadow
        - shadow_color_map: (H, W, 3) uint8 BGR shadow color, or None
    """
    # Find person bounding box
    rows = np.any(mask > 0.5, axis=1)
    cols = np.any(mask > 0.5, axis=0)
    if not rows.any() or not cols.any():
        return np.zeros((frame_h, frame_w), dtype=np.float32), None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    person_h = y_max - y_min
    person_w = x_max - x_min
    person_cx = (x_min + x_max) // 2
    feet_y = y_max  # Bottom of person = feet

    if person_h < 10 or person_w < 10:
        return np.zeros((frame_h, frame_w), dtype=np.float32), None

    # Apply light direction offset to shadow center
    shadow_cx = person_cx + int(person_w * light_offset_x)

    # Extract person region mask
    person_region = mask[y_min:y_max, x_min:x_max]

    # --- Layer 1: Contact shadow (sharp, close to feet) ---
    contact_h = max(3, int(person_h * 0.08))
    contact_w = int(person_w * 1.15)
    contact_squashed = cv2.resize(person_region, (contact_w, contact_h),
                                   interpolation=cv2.INTER_AREA)

    contact_canvas = np.zeros((frame_h, frame_w), dtype=np.float32)
    cx_start = shadow_cx - contact_w // 2
    cy_start = feet_y + int(frame_h * 0.005)  # Very close to feet

    # Clip and place contact shadow
    src_x1 = max(0, -cx_start)
    src_y1 = max(0, -cy_start)
    dst_x1 = max(0, cx_start)
    dst_y1 = max(0, cy_start)
    dst_x2 = min(frame_w, cx_start + contact_w)
    dst_y2 = min(frame_h, cy_start + contact_h)
    sx2 = src_x1 + (dst_x2 - dst_x1)
    sy2 = src_y1 + (dst_y2 - dst_y1)

    if dst_x2 > dst_x1 and dst_y2 > dst_y1 and sx2 <= contact_w and sy2 <= contact_h:
        contact_canvas[dst_y1:dst_y2, dst_x1:dst_x2] = \
            contact_squashed[src_y1:sy2, src_x1:sx2]

    # Mild blur for contact shadow (sharp but not pixel-hard)
    contact_blur = max(11, blur_size // 3)
    if contact_blur % 2 == 0:
        contact_blur += 1
    contact_canvas = cv2.GaussianBlur(contact_canvas, (contact_blur, contact_blur), 0)

    # --- Layer 2: Ambient shadow (diffused, wider) ---
    ambient_h = max(5, int(person_h * squash_factor))
    ambient_w = int(person_w * 1.4)
    ambient_squashed = cv2.resize(person_region, (ambient_w, ambient_h),
                                   interpolation=cv2.INTER_AREA)

    ambient_canvas = np.zeros((frame_h, frame_w), dtype=np.float32)
    ax_start = shadow_cx - ambient_w // 2
    ay_start = feet_y + int(frame_h * y_offset_ratio)

    src_x1 = max(0, -ax_start)
    src_y1 = max(0, -ay_start)
    dst_x1 = max(0, ax_start)
    dst_y1 = max(0, ay_start)
    dst_x2 = min(frame_w, ax_start + ambient_w)
    dst_y2 = min(frame_h, ay_start + ambient_h)
    sx2 = src_x1 + (dst_x2 - dst_x1)
    sy2 = src_y1 + (dst_y2 - dst_y1)

    if dst_x2 > dst_x1 and dst_y2 > dst_y1 and sx2 <= ambient_w and sy2 <= ambient_h:
        ambient_canvas[dst_y1:dst_y2, dst_x1:dst_x2] = \
            ambient_squashed[src_y1:sy2, src_x1:sx2]

    # Diffuse blur for ambient shadow
    if blur_size % 2 == 0:
        blur_size += 1
    ambient_canvas = cv2.GaussianBlur(ambient_canvas, (blur_size, blur_size), 0)

    # --- Combine layers ---
    # Contact shadow: 70% of total opacity (strong ground contact)
    # Ambient shadow: 50% of total opacity (soft spread)
    combined = np.clip(
        contact_canvas * shadow_opacity * 0.7 + ambient_canvas * shadow_opacity * 0.5,
        0.0, shadow_opacity
    )

    # Remove shadow from person area (shadow only on floor)
    combined = combined * (1.0 - mask)

    # Build color map if shadow color is provided
    color_map = None
    if shadow_color_bgr is not None:
        color_map = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        for c in range(3):
            color_map[:, :, c] = shadow_color_bgr[c]

    return combined, color_map


def add_shadow_to_video(input_path: str, shadow_opacity: float = 0.55) -> str:
    """
    Add projected floor shadow to a dance video.

    Analyzes background lighting on the first frame to determine shadow
    direction and color, then applies consistent shadow across all frames.

    Args:
        input_path: Path to input video.
        shadow_opacity: Shadow intensity (0.0-1.0).

    Returns:
        Path to output video (replaces input in-place).
    """
    segmentor = _create_segmentor()

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

    # Lighting analysis state (computed on first valid frame)
    light_offset_x = 0.0
    shadow_color = None
    lighting_analyzed = False

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe Tasks API expects RGB mp.Image
        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * 1000 / fps)
        result = segmentor.segment_for_video(mp_image, timestamp_ms=timestamp_ms)

        if result.confidence_masks and len(result.confidence_masks) > 0:
            mask = result.confidence_masks[0].numpy_view()  # (H, W, 1) float32
            if mask.ndim == 3:
                mask = mask[:, :, 0]  # squeeze to (H, W)

            # Analyze lighting on first frame with valid segmentation
            if not lighting_analyzed:
                light_offset_x, _, shadow_color = _analyze_light_source(frame, mask)
                lighting_analyzed = True
                logging.info(f"Shadow: Light analysis → offset_x={light_offset_x:.2f}, "
                             f"color=BGR({shadow_color[0]},{shadow_color[1]},{shadow_color[2]})")

            # Create floor shadow
            shadow_alpha, color_map = _create_floor_shadow(
                mask, h, w,
                shadow_opacity=shadow_opacity,
                blur_size=max(51, min(151, w // 8)),
                squash_factor=0.22,
                y_offset_ratio=0.01,
                light_offset_x=light_offset_x,
                shadow_color_bgr=shadow_color,
            )

            # Composite: blend shadow color with frame
            frame_float = frame.astype(np.float32)
            shadow_3ch = shadow_alpha[:, :, np.newaxis]

            if color_map is not None:
                # Tinted shadow: blend between frame and shadow color
                color_float = color_map.astype(np.float32)
                frame_float = frame_float * (1.0 - shadow_3ch) + color_float * shadow_3ch
            else:
                # Fallback: simple darkening
                frame_float = frame_float * (1.0 - shadow_3ch)

            frame = frame_float.clip(0, 255).astype(np.uint8)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            logging.info(f"Shadow: Processed {frame_idx}/{total_frames} frames")

    cap.release()
    writer.release()
    segmentor.close()
    logging.info(f"Shadow: Processing complete ({frame_idx} frames)")

    # Validate output: must have reasonable size (at least 10% of original)
    input_size = os.path.getsize(input_path)
    output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    if output_size < max(1024, input_size * 0.1):
        logging.warning(f"Shadow: Output too small ({output_size} bytes vs input {input_size} bytes), keeping original")
        if os.path.exists(output_path):
            os.remove(output_path)
        return input_path

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
