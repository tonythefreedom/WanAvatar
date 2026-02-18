"""
Camera motion extraction and background warping.

Extracts camera movement from the original dance video using feature matching
and homography estimation, then applies the same motion to a static background
image to produce a per-frame warped background video.

This solves the problem where a static background image stays fixed while
the original video has camera panning/tilting/tracking.
"""
import os
import math
import cv2
import numpy as np
import logging
import subprocess
import hashlib
from pathlib import Path


def _estimate_homographies(video_path: str, max_frames: int = 0,
                           force_rate: int = 16,
                           max_translation_px: float = 0.0) -> list:
    """
    Extract per-frame camera affine transforms from a video.

    Uses ORB feature detection on background regions (excluding the center
    where the dancer typically is), then estimates a partial affine transform
    (translation + rotation + uniform scale = 4 DOF) with RANSAC for stability.

    Per-frame transforms are validated: unreasonable translations, rotations,
    or scale changes are rejected. Cumulative translation is clamped to
    max_translation_px to prevent drift/divergence.

    Args:
        video_path: Path to the original dance video.
        max_frames: Max frames to process (0 = all).
        force_rate: Target FPS (to match ComfyUI resampling).
        max_translation_px: Max cumulative translation in pixels (0 = auto from video size).

    Returns:
        List of 3x3 cumulative homography matrices (frame 0 = identity).
    """
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if orig_fps <= 0 or total <= 0:
        cap.release()
        return []

    # Auto max translation: 25% of frame diagonal (increased for larger movements)
    if max_translation_px <= 0:
        max_translation_px = 0.25 * np.sqrt(w**2 + h**2)

    # Per-frame sanity limits (relaxed for fast zoom/pan)
    max_per_frame_tx = w * 0.15   # Max 15% of width per frame (increased from 5%)
    max_per_frame_ty = h * 0.15   # Max 15% of height per frame (increased from 5%)
    max_per_frame_angle = np.radians(5.0)  # Max 5 degrees per frame (increased from 2)
    min_per_frame_scale = 0.85    # Allow 15% zoom out per frame (was 0.95)
    max_per_frame_scale = 1.15    # Allow 15% zoom in per frame (was 1.05)

    # Frame sampling to match force_rate (ComfyUI resamples to 16fps)
    step = max(1, round(orig_fps / force_rate))

    # Create masks for feature detection
    
    # 1. Floor-focused mask (Primary strategy: track ground plane & background)
    # Good for full-body dance videos to prevent sliding feet
    floor_mask = np.ones((h, w), dtype=np.uint8) * 255
    x1_f = int(w * 0.30)
    x2_f = int(w * 0.70)
    y1_f = int(h * 0.15)
    y2_f = int(h * 0.85) # Exclude down to 85%, leaving bottom 15% for floor
    floor_mask[y1_f:y2_f, x1_f:x2_f] = 0
    
    # 2. Fallback mask (Secondary strategy: track general background)
    # Good for upper-body videos where floor is not visible or has no features
    fallback_mask = np.ones((h, w), dtype=np.uint8) * 255
    cx, cy = w // 2, h // 2
    margin_x = int(w * 0.20)
    margin_y = int(h * 0.15)
    fallback_mask[cy - margin_y:cy + margin_y, cx - margin_x:cx + margin_x] = 0
    
    logging.info("CameraMotion: Initialized floor mask and fallback mask")

    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    homographies = [np.eye(3, dtype=np.float64)]  # Frame 0 = identity
    cumulative_H = np.eye(3, dtype=np.float64)
    rejected = 0
    
    # Track mask strategy usage
    used_floor_mask = 0
    used_fallback_mask = 0

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return []
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_idx = 0
    sampled = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Skip frames to match target fps
        if frame_idx % step != 0:
            continue

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Strategy 1: Try floor mask first
        kp1, des1 = orb.detectAndCompute(prev_gray, floor_mask)
        kp2, des2 = orb.detectAndCompute(curr_gray, floor_mask)
        
        # Check if we have enough features on the floor/background
        min_features = 30
        if des1 is None or des2 is None or len(kp1) < min_features or len(kp2) < min_features:
            # Strategy 2: Fallback to general background mask
            kp1, des1 = orb.detectAndCompute(prev_gray, fallback_mask)
            kp2, des2 = orb.detectAndCompute(curr_gray, fallback_mask)
            used_fallback_mask += 1
        else:
            used_floor_mask += 1

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            homographies.append(cumulative_H.copy())
            prev_gray = curr_gray
            sampled += 1
            if max_frames > 0 and sampled >= max_frames - 1:
                break
            continue

        # Match features using KNN + ratio test
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m_list in matches:
            if len(m_list) == 2:
                m, n = m_list
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) < 8:
            homographies.append(cumulative_H.copy())
            prev_gray = curr_gray
            sampled += 1
            if max_frames > 0 and sampled >= max_frames - 1:
                break
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Use partial affine (4 DOF: translation + rotation + uniform scale)
        # More stable than full homography (8 DOF) for camera motion
        M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC,
                                                   ransacReprojThreshold=5.0)

        frame_valid = False
        if M is not None:
            # Convert 2x3 affine to 3x3 homography
            H = np.eye(3, dtype=np.float64)
            H[:2, :] = M

            # Validate per-frame transform
            tx, ty = H[0, 2], H[1, 2]
            angle = np.arctan2(H[1, 0], H[0, 0])
            scale = np.sqrt(H[0, 0]**2 + H[1, 0]**2)

            if (abs(tx) <= max_per_frame_tx and
                abs(ty) <= max_per_frame_ty and
                abs(angle) <= max_per_frame_angle and
                min_per_frame_scale <= scale <= max_per_frame_scale):
                cumulative_H = H @ cumulative_H
                frame_valid = True
            else:
                rejected += 1

        # Clamp cumulative translation to prevent drift
        cum_tx = cumulative_H[0, 2]
        cum_ty = cumulative_H[1, 2]
        if abs(cum_tx) > max_translation_px:
            cumulative_H[0, 2] = np.sign(cum_tx) * max_translation_px
        if abs(cum_ty) > max_translation_px:
            cumulative_H[1, 2] = np.sign(cum_ty) * max_translation_px

        homographies.append(cumulative_H.copy())
        prev_gray = curr_gray
        sampled += 1
        if max_frames > 0 and sampled >= max_frames - 1:
            break

    cap.release()
    logging.info(f"CameraMotion: Extracted {len(homographies)} transforms "
                 f"from {frame_idx + 1} frames (step={step}, "
                 f"rejected={rejected}, max_drift={max_translation_px:.0f}px). "
                 f"Mask usage: Floor={used_floor_mask}, Fallback={used_fallback_mask}")
    return homographies


def _smooth_homographies(homographies: list, window: int = 5) -> list:
    """
    Smooth homographies using a moving average window to reduce jitter.
    
    Args:
        homographies: List of 3x3 homography matrices
        window: Window size for moving average (odd number recommended)
    
    Returns:
        Smoothed list of homography matrices
    """
    if len(homographies) < 2:
        return homographies
    
    smoothed = []
    half_window = window // 2
    
    for i in range(len(homographies)):
        start = max(0, i - half_window)
        end = min(len(homographies), i + half_window + 1)
        window_matrices = homographies[start:end]
        
        # Average the matrices
        avg_H = np.mean([H for H in window_matrices], axis=0)
        smoothed.append(avg_H)
    
    return smoothed


# ---------------------------------------------------------------------------
# Dancer position tracking (MediaPipe Pose)
# ---------------------------------------------------------------------------

_POSE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "models", "pose_landmarker_lite.task")

# MediaPipe Pose landmark indices
_LEFT_ANKLE = 27
_RIGHT_ANKLE = 28
_LEFT_HIP = 23
_RIGHT_HIP = 24
_VISIBILITY_THRESHOLD = 0.5


def _get_dancer_center_x(result) -> float | None:
    """Extract dancer's center X (normalized 0-1) from PoseLandmarkerResult.

    Primary: midpoint of left/right ankles.
    Fallback: midpoint of left/right hips.
    Returns None if no pose detected or landmarks below visibility threshold.
    """
    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return None

    lm = result.pose_landmarks[0]

    la, ra = lm[_LEFT_ANKLE], lm[_RIGHT_ANKLE]
    if la.visibility >= _VISIBILITY_THRESHOLD and ra.visibility >= _VISIBILITY_THRESHOLD:
        return (la.x + ra.x) / 2.0

    lh, rh = lm[_LEFT_HIP], lm[_RIGHT_HIP]
    if lh.visibility >= _VISIBILITY_THRESHOLD and rh.visibility >= _VISIBILITY_THRESHOLD:
        return (lh.x + rh.x) / 2.0

    for landmark in [la, ra, lh, rh]:
        if landmark.visibility >= _VISIBILITY_THRESHOLD:
            return landmark.x

    return None


def _interpolate_gaps(positions: list) -> list:
    """Fill None values via linear interpolation; edge Nones use nearest valid."""
    n = len(positions)
    if n == 0:
        return []
    result = list(positions)
    valid = [i for i, p in enumerate(result) if p is not None]
    if not valid:
        return [0.5] * n

    for i in range(valid[0]):
        result[i] = result[valid[0]]
    for i in range(valid[-1] + 1, n):
        result[i] = result[valid[-1]]

    for idx in range(len(valid) - 1):
        a, b = valid[idx], valid[idx + 1]
        if b - a > 1:
            for k in range(a + 1, b):
                t = (k - a) / (b - a)
                result[k] = result[a] * (1 - t) + result[b] * t
    return result


def _smooth_dancer_positions(positions: list, window: int = 7) -> list:
    """Moving average smoothing for dancer positions."""
    if len(positions) < 2:
        return positions
    half = window // 2
    smoothed = []
    for i in range(len(positions)):
        start = max(0, i - half)
        end = min(len(positions), i + half + 1)
        smoothed.append(float(np.mean(positions[start:end])))
    return smoothed


def _extract_dancer_positions(video_path: str, force_rate: int = 16,
                              max_frames: int = 0,
                              smooth_window: int = 7) -> list:
    """Extract dancer's horizontal center position (normalized 0-1) per sampled frame.

    Uses MediaPipe Pose Landmarker to detect ankle positions and compute
    the midpoint X coordinate.  Falls back to hip midpoint when ankles are
    not visible.

    Returns:
        List of normalized X positions (0.0=left, 1.0=right), one per
        sampled frame.  Empty list on failure.
    """
    if not os.path.exists(_POSE_MODEL_PATH):
        logging.warning(f"DancerTrack: Pose model not found: {_POSE_MODEL_PATH}")
        return []

    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python import BaseOptions
    except ImportError:
        logging.warning("DancerTrack: mediapipe not installed")
        return []

    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if orig_fps <= 0 or total <= 0:
        cap.release()
        return []

    step = max(1, round(orig_fps / force_rate))

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_POSE_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.3,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    raw: list[float | None] = []
    frame_idx = 0

    ret, frame = cap.read()
    if not ret:
        cap.release()
        landmarker.close()
        return []

    # Frame 0
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts = int(frame_idx * 1000 / orig_fps)
    raw.append(_get_dancer_center_x(landmarker.detect_for_video(mp_img, timestamp_ms=ts)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % step != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(frame_idx * 1000 / orig_fps)
        raw.append(_get_dancer_center_x(landmarker.detect_for_video(mp_img, timestamp_ms=ts)))

        if max_frames > 0 and len(raw) >= max_frames:
            break

    cap.release()
    landmarker.close()

    detected = sum(1 for p in raw if p is not None)
    positions = _interpolate_gaps(raw)
    positions = _smooth_dancer_positions(positions, window=smooth_window)

    logging.info(f"DancerTrack: {len(positions)} frames, {detected}/{len(raw)} detected, step={step}")
    return positions


def outpaint_background(bg_image_path: str, target_width: int, target_height: int,
                        scale_factor: float = 1.6, cache_dir: str = "/tmp/bg_outpaint_cache") -> str:
    """
    Outpaint a background image to provide room for camera motion without quality loss.
    
    New logic:
    1. If background >= target resolution: Keep original size, only add margins for camera motion
    2. If background < target resolution: Add pixels to reach target size + margins
    3. Never downscale the background
    
    Args:
        bg_image_path: Path to the original background image.
        target_width: Target video width (e.g., 720 for portrait).
        target_height: Target video height (e.g., 1280 for portrait).
        scale_factor: How much larger to make the canvas (1.3 = 30% margin).
        cache_dir: Directory to cache outpainted results.
    
    Returns:
        Path to the outpainted background image, or original path if outpainting fails.
    """
    if not os.path.exists(bg_image_path):
        logging.warning(f"Outpaint: Background image not found: {bg_image_path}")
        return bg_image_path
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key from image content + target dimensions
    with open(bg_image_path, 'rb') as f:
        img_hash = hashlib.md5(f.read()).hexdigest()[:16]
    cache_key = f"{img_hash}_{target_width}x{target_height}_s{scale_factor:.2f}"
    cache_path = os.path.join(cache_dir, f"{cache_key}.png")
    
    # Return cached result if exists
    if os.path.exists(cache_path):
        logging.info(f"Outpaint: Using cached result: {cache_path}")
        return cache_path
    
    # Load original image
    bg_img = cv2.imread(bg_image_path)
    if bg_img is None:
        logging.warning(f"Outpaint: Failed to read image: {bg_image_path}")
        return bg_image_path
    
    orig_h, orig_w = bg_img.shape[:2]
    
    # Calculate required final dimensions for camera motion
    required_w = int(target_width * scale_factor)
    required_h = int(target_height * scale_factor)
    target_aspect = target_width / target_height  # Use target video aspect ratio (e.g., 720/1280 = 0.5625)
    orig_aspect = orig_w / orig_h
    
    # Check if aspect ratio is already correct (within 2% tolerance)
    aspect_diff = abs(orig_aspect - target_aspect) / target_aspect
    
    # Case 1: Background is large enough (can downscale) → Crop to target aspect + resize to required
    if orig_w >= required_w and orig_h >= required_h:
        logging.info(f"Outpaint: Background {orig_w}x{orig_h} >= required {required_w}x{required_h}, cropping to target aspect ratio")
        
        # Calculate crop dimensions to match TARGET aspect ratio (not required aspect)
        if orig_aspect > target_aspect:
            # Background is wider → crop width
            crop_h = orig_h
            crop_w = int(crop_h * target_aspect)
        else:
            # Background is taller → crop height
            crop_w = orig_w
            crop_h = int(crop_w / target_aspect)
        
        # Center crop
        x = (orig_w - crop_w) // 2
        y = (orig_h - crop_h) // 2
        cropped = bg_img[y:y+crop_h, x:x+crop_w]
        
        # Resize to required dimensions (downscale with camera motion margin)
        resized = cv2.resize(cropped, (required_w, required_h), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(cache_path, resized)
        
        logging.info(f"Outpaint: Cropped {orig_w}x{orig_h} → {crop_w}x{crop_h} (aspect {crop_w/crop_h:.3f}) → resized to {required_w}x{required_h}")
        return cache_path
    
    # Case 2: Background is too small → Need outpainting (never upscale with resize)
    # First, ensure correct aspect ratio, then extend to required size
    
    if aspect_diff >= 0.01:  # 1% tolerance for aspect ratio
        # Aspect ratio is wrong → First crop to correct TARGET aspect ratio
        logging.info(f"Outpaint: Adjusting aspect ratio from {orig_aspect:.3f} to target {target_aspect:.3f} (diff={aspect_diff*100:.1f}%)")
        
        if orig_aspect > target_aspect:
            # Too wide → crop width
            crop_h = orig_h
            crop_w = int(crop_h * target_aspect)
        else:
            # Too tall → crop height
            crop_w = orig_w
            crop_h = int(crop_w / target_aspect)
        
        # Center crop
        x = (orig_w - crop_w) // 2
        y = (orig_h - crop_h) // 2
        bg_img = bg_img[y:y+crop_h, x:x+crop_w]
        orig_w, orig_h = crop_w, crop_h
        logging.info(f"Outpaint: Cropped to target aspect ratio: {orig_w}x{orig_h} (aspect {orig_w/orig_h:.3f})")
    
    # Now extend to required size with outpainting
    final_w = required_w
    final_h = required_h
    
    # Calculate margins to add
    margin_w = (final_w - orig_w) // 2
    margin_h = (final_h - orig_h) // 2
    
    logging.info(f"Outpaint: Extending {orig_w}x{orig_h}, add margins ({margin_w}px H, {margin_h}px V) → {final_w}x{final_h}")
    
    try:
        # Method 1: Use FLUX.2 for AI outpainting
        outpainted_path = _outpaint_with_flux_v2(bg_img, final_w, final_h, margin_w, margin_h, cache_path)
        if outpainted_path:
            return outpainted_path
    except Exception as e:
        logging.warning(f"Outpaint: FLUX method failed: {e}")
    
    try:
        # Method 2: Fallback to intelligent padding
        outpainted_path = _outpaint_with_padding_v2(bg_img, final_w, final_h, margin_w, margin_h, cache_path)
        if outpainted_path:
            return outpainted_path
    except Exception as e:
        logging.warning(f"Outpaint: Padding method failed: {e}")
    
    # Method 3: Final fallback - simple edge extension (better than returning original)
    try:
        logging.info(f"Outpaint: Using edge extension fallback")
        canvas = np.zeros((final_h, final_w, 3), dtype=np.uint8)
        
        # Place original image in center
        y_offset = (final_h - orig_h) // 2
        x_offset = (final_w - orig_w) // 2
        canvas[y_offset:y_offset+orig_h, x_offset:x_offset+orig_w] = bg_img
        
        # Fill margins by extending edges
        # Top margin
        if y_offset > 0:
            canvas[:y_offset, x_offset:x_offset+orig_w] = bg_img[0:1, :]
        # Bottom margin
        if y_offset + orig_h < final_h:
            canvas[y_offset+orig_h:, x_offset:x_offset+orig_w] = bg_img[-1:, :]
        # Left margin
        if x_offset > 0:
            canvas[:, :x_offset] = canvas[:, x_offset:x_offset+1]
        # Right margin
        if x_offset + orig_w < final_w:
            canvas[:, x_offset+orig_w:] = canvas[:, x_offset+orig_w-1:x_offset+orig_w]
        
        cv2.imwrite(cache_path, canvas)
        logging.info(f"Outpaint: Edge extension successful → {cache_path}")
        return cache_path
    except Exception as e:
        logging.warning(f"Outpaint: Edge extension failed: {e}")
    
    # If all methods fail, return original
    logging.warning(f"Outpaint: All methods failed, using original image")
    return bg_image_path


def _outpaint_with_flux_v2(bg_original: np.ndarray, final_w: int, final_h: int, 
                           margin_w: int, margin_h: int, output_path: str) -> str:
    """
    V2: AI-powered outpainting using FLUX.1 Fill [dev].
    
    Args:
        bg_original: Original background image (keep at original resolution)
        final_w: Final width with margins
        final_h: Final height with margins
        margin_w: Horizontal margin on each side
        margin_h: Vertical margin on each side
        output_path: Where to save result
    
    Returns:
        Path to outpainted image, or empty string on failure.
    """
    try:
        from diffusers import FluxFillPipeline
        import torch
        from PIL import Image
        import numpy as np
        
        # Clear GPU memory before loading FLUX
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logging.info(f"FLUX Outpaint: Loading FLUX.1 Fill [dev]...")
        
        # Load pipeline from Hugging Face
        pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            torch_dtype=torch.bfloat16,
            cache_dir="/mnt/models",
        ).to("cuda:0")
        
        orig_h, orig_w = bg_original.shape[:2]
        
        # Step 1: Create canvas with intelligent padding (mirror edges)
        canvas_np = np.zeros((final_h, final_w, 3), dtype=np.uint8)
        
        # Fill edges with mirrored content for better context
        if margin_h > 0:
            # Top edge - mirror top rows
            edge_size = min(margin_h, orig_h // 4)
            edge = bg_original[:edge_size, :]
            edge_flipped = cv2.flip(edge, 0)
            edge_resized = cv2.resize(edge_flipped, (orig_w, margin_h), interpolation=cv2.INTER_CUBIC)
            canvas_np[:margin_h, margin_w:margin_w+orig_w] = edge_resized
            
            # Bottom edge - mirror bottom rows
            edge = bg_original[-edge_size:, :]
            edge_flipped = cv2.flip(edge, 0)
            edge_resized = cv2.resize(edge_flipped, (orig_w, margin_h), interpolation=cv2.INTER_CUBIC)
            canvas_np[margin_h+orig_h:, margin_w:margin_w+orig_w] = edge_resized
        
        if margin_w > 0:
            # Left edge - mirror left columns
            edge_size = min(margin_w, orig_w // 4)
            edge = bg_original[:, :edge_size]
            edge_flipped = cv2.flip(edge, 1)
            edge_resized = cv2.resize(edge_flipped, (margin_w, orig_h), interpolation=cv2.INTER_CUBIC)
            canvas_np[margin_h:margin_h+orig_h, :margin_w] = edge_resized
            
            # Right edge - mirror right columns
            edge = bg_original[:, -edge_size:]
            edge_flipped = cv2.flip(edge, 1)
            edge_resized = cv2.resize(edge_flipped, (margin_w, orig_h), interpolation=cv2.INTER_CUBIC)
            canvas_np[margin_h:margin_h+orig_h, margin_w+orig_w:] = edge_resized
        
        # Fill corners with blurred edge content
        if margin_h > 0 and margin_w > 0:
            corner_size = 20
            # Top-left
            corner = bg_original[:corner_size, :corner_size]
            corner_blurred = cv2.GaussianBlur(corner, (21, 21), 0)
            canvas_np[:margin_h, :margin_w] = cv2.resize(corner_blurred, (margin_w, margin_h), interpolation=cv2.INTER_CUBIC)
            # Top-right
            corner = bg_original[:corner_size, -corner_size:]
            corner_blurred = cv2.GaussianBlur(corner, (21, 21), 0)
            canvas_np[:margin_h, margin_w+orig_w:] = cv2.resize(corner_blurred, (margin_w, margin_h), interpolation=cv2.INTER_CUBIC)
            # Bottom-left
            corner = bg_original[-corner_size:, :corner_size]
            corner_blurred = cv2.GaussianBlur(corner, (21, 21), 0)
            canvas_np[margin_h+orig_h:, :margin_w] = cv2.resize(corner_blurred, (margin_w, margin_h), interpolation=cv2.INTER_CUBIC)
            # Bottom-right
            corner = bg_original[-corner_size:, -corner_size:]
            corner_blurred = cv2.GaussianBlur(corner, (21, 21), 0)
            canvas_np[margin_h+orig_h:, margin_w+orig_w:] = cv2.resize(corner_blurred, (margin_w, margin_h), interpolation=cv2.INTER_CUBIC)
        
        # Place original in center
        canvas_np[margin_h:margin_h+orig_h, margin_w:margin_w+orig_w] = bg_original
        
        # Step 2: Create soft mask for smooth transitions
        mask_np = np.ones((final_h, final_w), dtype=np.uint8) * 255
        
        # Create feathered mask (gradually transition from 0 to 255)
        feather_size = min(40, margin_w, margin_h) if margin_w > 0 and margin_h > 0 else 40
        
        # Center region is fully masked out (keep original)
        inner_margin_h = margin_h + feather_size if margin_h > 0 else feather_size
        inner_margin_w = margin_w + feather_size if margin_w > 0 else feather_size
        
        if inner_margin_h < final_h // 2 and inner_margin_w < final_w // 2:
            mask_np[inner_margin_h:final_h-inner_margin_h, 
                    inner_margin_w:final_w-inner_margin_w] = 0
        
        # Apply Gaussian blur to mask for smooth transitions
        mask_np = cv2.GaussianBlur(mask_np, (51, 51), 0)
        
        # Convert to PIL
        canvas_pil = Image.fromarray(cv2.cvtColor(canvas_np, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask_np)
        
        # Step 3: Use FLUX.1 Fill to refine the intelligent padding
        prompt = "continue the background naturally, maintain architectural perspective and lighting"
        
        logging.info(f"FLUX Outpaint: Refining edges with FLUX.1 Fill...")
        
        generated = pipe(
            image=canvas_pil,
            mask_image=mask_pil,
            prompt=prompt,
            height=final_h,
            width=final_w,
            num_inference_steps=50,
            guidance_scale=30.0,
            generator=torch.Generator("cuda:0").manual_seed(42),
        ).images[0]
        
        # Convert result
        generated_np = np.array(generated)
        result_np = cv2.cvtColor(generated_np, cv2.COLOR_RGB2BGR)
        
        # FLUX may resize to nearest multiple of 16, so resize back if needed
        gen_h, gen_w = result_np.shape[:2]
        if gen_h != final_h or gen_w != final_w:
            logging.info(f"FLUX Outpaint: Resizing generated image from {gen_w}x{gen_h} to {final_w}x{final_h}")
            result_np = cv2.resize(result_np, (final_w, final_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Save result
        cv2.imwrite(output_path, result_np)
        
        # Clean up
        del pipe
        torch.cuda.empty_cache()
        
        logging.info(f"FLUX Outpaint: AI generation complete, saved to {output_path}")
        return output_path
        
    except Exception as e:
        logging.warning(f"FLUX Outpaint failed: {e}, using padding fallback")
        import traceback
        traceback.print_exc()
        return ""


def _outpaint_with_padding_v2(bg_original: np.ndarray, final_w: int, final_h: int,
                               margin_w: int, margin_h: int, output_path: str) -> str:
    """
    V2: Intelligent padding fallback (no FLUX).
    
    Args:
        bg_original: Original background image
        final_w: Final width with margins
        final_h: Final height with margins
        margin_w: Horizontal margin on each side
        margin_h: Vertical margin on each side
        output_path: Where to save result
    
    Returns:
        Path to outpainted image, or empty string on failure.
    """
    try:
        orig_h, orig_w = bg_original.shape[:2]
        
        # Create canvas
        canvas = np.zeros((final_h, final_w, 3), dtype=np.uint8)
        
        # Use intelligent edge mirroring and blending
        blend_width = 50
        
        # Fill edges with mirrored and blended content
        if margin_h > 0:
            # Top
            edge = bg_original[:min(margin_h, orig_h), :]
            edge_flipped = cv2.flip(edge, 0)
            edge_resized = cv2.resize(edge_flipped, (orig_w, margin_h), interpolation=cv2.INTER_CUBIC)
            canvas[:margin_h, margin_w:margin_w+orig_w] = edge_resized
            
            # Blend top edge
            for i in range(min(blend_width, margin_h, orig_h // 4)):
                alpha = i / min(blend_width, margin_h, orig_h // 4)
                canvas[margin_h - i - 1, margin_w:margin_w+orig_w] = (
                    edge_resized[margin_h - i - 1, :] * (1 - alpha) +
                    bg_original[0, :] * alpha
                ).astype(np.uint8)
            
            # Bottom
            edge = bg_original[-min(margin_h, orig_h):, :]
            edge_flipped = cv2.flip(edge, 0)
            edge_resized = cv2.resize(edge_flipped, (orig_w, margin_h), interpolation=cv2.INTER_CUBIC)
            canvas[margin_h+orig_h:, margin_w:margin_w+orig_w] = edge_resized
            
            # Blend bottom edge
            for i in range(min(blend_width, margin_h, orig_h // 4)):
                alpha = i / min(blend_width, margin_h, orig_h // 4)
                if margin_h + orig_h + i < final_h:
                    canvas[margin_h + orig_h + i, margin_w:margin_w+orig_w] = (
                        edge_resized[i, :] * (1 - alpha) +
                        bg_original[-1, :] * alpha
                    ).astype(np.uint8)
        
        if margin_w > 0:
            # Left
            edge = bg_original[:, :min(margin_w, orig_w)]
            edge_flipped = cv2.flip(edge, 1)
            edge_resized = cv2.resize(edge_flipped, (margin_w, orig_h), interpolation=cv2.INTER_CUBIC)
            canvas[margin_h:margin_h+orig_h, :margin_w] = edge_resized
            
            # Blend left edge
            for i in range(min(blend_width, margin_w, orig_w // 4)):
                alpha = i / min(blend_width, margin_w, orig_w // 4)
                canvas[margin_h:margin_h+orig_h, margin_w - i - 1] = (
                    edge_resized[:, margin_w - i - 1] * (1 - alpha) +
                    bg_original[:, 0] * alpha
                ).astype(np.uint8)
            
            # Right
            edge = bg_original[:, -min(margin_w, orig_w):]
            edge_flipped = cv2.flip(edge, 1)
            edge_resized = cv2.resize(edge_flipped, (margin_w, orig_h), interpolation=cv2.INTER_CUBIC)
            canvas[margin_h:margin_h+orig_h, margin_w+orig_w:] = edge_resized
            
            # Blend right edge
            for i in range(min(blend_width, margin_w, orig_w // 4)):
                alpha = i / min(blend_width, margin_w, orig_w // 4)
                if margin_w + orig_w + i < final_w:
                    canvas[margin_h:margin_h+orig_h, margin_w + orig_w + i] = (
                        edge_resized[:, i] * (1 - alpha) +
                        bg_original[:, -1] * alpha
                    ).astype(np.uint8)
        
        # Fill corners
        if margin_h > 0 and margin_w > 0:
            canvas[:margin_h, :margin_w] = cv2.resize(
                bg_original[:min(margin_h, orig_h), :min(margin_w, orig_w)],
                (margin_w, margin_h), interpolation=cv2.INTER_CUBIC
            )
            canvas[:margin_h, margin_w+orig_w:] = cv2.resize(
                bg_original[:min(margin_h, orig_h), -min(margin_w, orig_w):],
                (margin_w, margin_h), interpolation=cv2.INTER_CUBIC
            )
            canvas[margin_h+orig_h:, :margin_w] = cv2.resize(
                bg_original[-min(margin_h, orig_h):, :min(margin_w, orig_w)],
                (margin_w, margin_h), interpolation=cv2.INTER_CUBIC
            )
            canvas[margin_h+orig_h:, margin_w+orig_w:] = cv2.resize(
                bg_original[-min(margin_h, orig_h):, -min(margin_w, orig_w):],
                (margin_w, margin_h), interpolation=cv2.INTER_CUBIC
            )
        
        # Place original in center
        canvas[margin_h:margin_h+orig_h, margin_w:margin_w+orig_w] = bg_original
        
        # Save
        cv2.imwrite(output_path, canvas)
        logging.info(f"Outpaint: Intelligent padding complete, saved to {output_path}")
        return output_path
        
    except Exception as e:
        logging.error(f"Padding outpaint failed: {e}")
        import traceback
        traceback.print_exc()
        return ""


def warp_background_video(bg_image_path: str, video_path: str,
                          output_path: str, force_rate: int = 16,
                          scale_factor: float = 1.6,
                          target_width: int = None,
                          target_height: int = None) -> str:
    """
    Create a warped background video that follows camera motion from the
    original dance video.

    Process:
    1. Extract camera motion (homographies) from original video
    2. Scale up background image to allow room for camera panning
    3. Apply inverse homography per frame to simulate camera following
    4. Save as H.264 MP4 video

    Args:
        bg_image_path: Path to the static background image.
        video_path: Path to the original dance video.
        output_path: Path to save the warped background video.
        force_rate: Target FPS (matches ComfyUI force_rate).
        scale_factor: How much to scale up bg image (1.3 = 30% larger).
        target_width: Target width for output video (overrides video dimensions).
        target_height: Target height for output video (overrides video dimensions).

    Returns:
        Path to the warped background video, or empty string on failure.
    """
    if not os.path.exists(bg_image_path):
        logging.warning(f"CameraMotion: Background image not found: {bg_image_path}")
        return ""
    if not os.path.exists(video_path):
        logging.warning(f"CameraMotion: Video not found: {video_path}")
        return ""

    # Read video dimensions
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if orig_fps <= 0 or vid_w <= 0 or vid_h <= 0:
        logging.warning(f"CameraMotion: Invalid video")
        return ""

    # Use target dimensions if provided, otherwise use video dimensions
    if target_width and target_height:
        vid_w, vid_h = target_width, target_height
        logging.info(f"CameraMotion: Using target dimensions {vid_w}x{vid_h}")

    # Calculate expected frame count after resampling to force_rate
    # Use ceil + buffer to ensure bg video is always >= dance video frames
    # ComfyUI's VHS_LoadVideo may produce slightly more frames than int() estimate
    duration = total_frames / orig_fps
    expected_frames = math.ceil(duration * force_rate) + 10
    expected_frames = max(expected_frames, 1)

    logging.info(f"CameraMotion: Video {vid_w}x{vid_h}, {orig_fps:.1f}fps, "
                 f"{duration:.1f}s → {expected_frames} frames at {force_rate}fps")

    # Step 1: Extract camera motion
    homographies = _estimate_homographies(video_path, max_frames=expected_frames,
                                          force_rate=force_rate)

    if len(homographies) < 2:
        logging.info("CameraMotion: No camera motion detected, skipping warp")
        return ""

    # Check if there's actually significant camera motion
    max_tx = max(abs(H[0, 2]) for H in homographies)
    max_ty = max(abs(H[1, 2]) for H in homographies)
    if max_tx < 5.0 and max_ty < 5.0:
        logging.info(f"CameraMotion: Minimal motion (max_tx={max_tx:.1f}, "
                     f"max_ty={max_ty:.1f}), skipping warp")
        return ""

    logging.info(f"CameraMotion: Max motion tx={max_tx:.1f}px, ty={max_ty:.1f}px")

    # Smooth to reduce jitter
    homographies = _smooth_homographies(homographies, window=7)

    # Step 1b: Extract dancer horizontal positions for platform alignment
    dancer_positions = _extract_dancer_positions(
        video_path, force_rate=force_rate,
        max_frames=expected_frames, smooth_window=7)
    use_dancer_tracking = len(dancer_positions) >= 2
    if use_dancer_tracking:
        while len(dancer_positions) < expected_frames:
            dancer_positions.append(dancer_positions[-1])
        ref_dancer_x = dancer_positions[0]
        dancer_range_px = (max(dancer_positions) - min(dancer_positions)) * vid_w
        logging.info(f"DancerTrack: Enabled. ref_x={ref_dancer_x:.3f}, "
                     f"range={dancer_range_px:.0f}px")
    else:
        dancer_range_px = 0
        logging.info("DancerTrack: Disabled (no pose data). Using camera motion pan.")

    # Step 2: Prepare background image with outpainting for better quality
    target_w, target_h = vid_w, vid_h

    # Dynamically adjust scale_factor to ensure camera motion fits within margins
    # Required margin = max_motion + safety buffer (20%)
    required_margin_x = max_tx * 1.2
    required_margin_y = max_ty * 1.2
    # Also account for dancer horizontal range if tracking is active
    if use_dancer_tracking:
        required_margin_x = max(required_margin_x, dancer_range_px / 2 * 1.2)

    # Calculate minimum scale_factor needed
    # margin_x = (needed_w - target_w) / 2 >= required_margin_x
    # needed_w = target_w + 2 * required_margin_x
    # scale_factor = needed_w / target_w
    min_scale_x = 1.0 + (2 * required_margin_x / target_w)
    min_scale_y = 1.0 + (2 * required_margin_y / target_h)
    min_scale = max(min_scale_x, min_scale_y)
    
    # Use the larger of default scale_factor or dynamically calculated minimum
    final_scale_factor = max(scale_factor, min_scale)
    
    if final_scale_factor > scale_factor:
        logging.info(f"CameraMotion: Adjusting scale_factor {scale_factor:.2f} → {final_scale_factor:.2f} to fit camera motion")
    
    needed_w = int(target_w * final_scale_factor)
    needed_h = int(target_h * final_scale_factor)
    
    # Use outpainting to expand background (preserves center quality)
    # Pass the final scale_factor so outpainting produces the exact size we need
    outpainted_bg_path = outpaint_background(bg_image_path, target_w, target_h, final_scale_factor)
    
    # Load the outpainted/processed background
    bg_img = cv2.imread(outpainted_bg_path)
    if bg_img is None:
        logging.warning(f"CameraMotion: Failed to read processed bg image")
        return ""
    
    bg_h, bg_w = bg_img.shape[:2]
    
    # Check if outpainted image already has the exact dimensions we need
    if bg_w == needed_w and bg_h == needed_h:
        bg_scaled = bg_img
        logging.info(f"CameraMotion: Using outpainted bg at exact size {bg_w}x{bg_h}")
    else:
        # Outpainted image size doesn't match - need to resize/crop
        # This should rarely happen if outpaint_background works correctly
        logging.warning(f"CameraMotion: Outpainted bg {bg_w}x{bg_h} != needed {needed_w}x{needed_h}, adjusting...")
        
        bg_aspect = bg_w / bg_h
        needed_aspect = needed_w / needed_h
        
        if abs(bg_aspect - needed_aspect) < 0.01:  # Aspect ratios match (within 1%)
            # Direct resize
            bg_scaled = cv2.resize(bg_img, (needed_w, needed_h), interpolation=cv2.INTER_LANCZOS4)
            logging.info(f"CameraMotion: Resized outpainted bg {bg_w}x{bg_h} → {needed_w}x{needed_h}")
        else:
            # Aspect ratios differ - resize to fit and center crop
            if bg_aspect > needed_aspect:
                # Background is wider - fit height, crop width
                scale = needed_h / bg_h
                temp_w = int(bg_w * scale)
                temp_h = needed_h
            else:
                # Background is taller - fit width, crop height
                scale = needed_w / bg_w
                temp_w = needed_w
                temp_h = int(bg_h * scale)
            
            temp_bg = cv2.resize(bg_img, (temp_w, temp_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Center crop to exact dimensions
            crop_x = (temp_w - needed_w) // 2
            crop_y = (temp_h - needed_h) // 2
            bg_scaled = temp_bg[crop_y:crop_y+needed_h, crop_x:crop_x+needed_w]
            logging.info(f"CameraMotion: Resized {bg_w}x{bg_h} → {temp_w}x{temp_h}, center crop → {needed_w}x{needed_h}")

    # Center offset (the initial view is the center crop of the scaled image)
    scaled_h, scaled_w = bg_scaled.shape[:2]
    offset_x = (scaled_w - target_w) // 2
    offset_y = (scaled_h - target_h) // 2

    # Step 3: Warp background per frame
    # Pre-compute the translation matrix to center the view
    T_center = np.array([
        [1, 0, -offset_x],
        [0, 1, -offset_y],
        [0, 0, 1]
    ], dtype=np.float64)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, force_rate, (target_w, target_h))

    # Match frame count: if homographies < expected_frames, pad with last H
    while len(homographies) < expected_frames:
        homographies.append(homographies[-1].copy())

    for i in range(expected_frames):
        H = homographies[i]

        # Vertical pan: disabled — dance videos have no meaningful Y-axis camera motion.
        # Any detected Y movement is noise from dancer motion being misread as camera pan.
        cx, cy = target_w / 2.0, target_h / 2.0
        mapped = H @ np.array([cx, cy, 1.0])
        mapped /= mapped[2]
        pan_dy = 0.0

        # Horizontal pan: dancer tracking (platform follows feet) or camera motion fallback
        if use_dancer_tracking:
            pan_dx = (dancer_positions[i] - ref_dancer_x) * target_w
        else:
            pan_dx = mapped[0] - cx

        warp_H = np.array([
            [1, 0, -offset_x + pan_dx],
            [0, 1, -offset_y + pan_dy],
            [0, 0, 1]
        ], dtype=np.float64)

        warped = cv2.warpPerspective(
            bg_scaled, warp_H, (target_w, target_h),
            borderMode=cv2.BORDER_REFLECT_101
        )
        writer.write(warped)

    writer.release()
    logging.info(f"CameraMotion: Wrote {expected_frames} warped frames to {output_path}")

    # Re-encode to H.264 for ComfyUI compatibility
    h264_path = output_path.replace(".mp4", "_h264.mp4")
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", output_path,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-an",
            h264_path,
        ], capture_output=True, timeout=120)
        if result.returncode == 0 and os.path.exists(h264_path) and os.path.getsize(h264_path) > 0:
            os.replace(h264_path, output_path)
            logging.info(f"CameraMotion: H.264 re-encoded: {output_path}")
        else:
            logging.warning(f"CameraMotion: H.264 re-encode failed, keeping mp4v")
    except Exception as e:
        logging.warning(f"CameraMotion: H.264 re-encode error: {e}")
    finally:
        if os.path.exists(h264_path) and h264_path != output_path:
            try:
                os.remove(h264_path)
            except OSError:
                pass

    return output_path
