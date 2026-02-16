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

    # Auto max translation: 15% of frame diagonal (matches scale_factor=1.3 margin)
    if max_translation_px <= 0:
        max_translation_px = 0.15 * np.sqrt(w**2 + h**2)

    # Per-frame sanity limits
    max_per_frame_tx = w * 0.05   # Max 5% of width per frame
    max_per_frame_ty = h * 0.05   # Max 5% of height per frame
    max_per_frame_angle = np.radians(2.0)  # Max 2 degrees per frame
    min_per_frame_scale = 0.95
    max_per_frame_scale = 1.05

    # Frame sampling to match force_rate (ComfyUI resamples to 16fps)
    step = max(1, round(orig_fps / force_rate))

    # Create mask to exclude center person area (keep edges for camera motion)
    bg_mask = np.ones((h, w), dtype=np.uint8) * 255
    cx, cy = w // 2, h // 2
    margin_x = int(w * 0.20)
    margin_y = int(h * 0.15)
    bg_mask[cy - margin_y:cy + margin_y, cx - margin_x:cx + margin_x] = 0

    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    homographies = [np.eye(3, dtype=np.float64)]  # Frame 0 = identity
    cumulative_H = np.eye(3, dtype=np.float64)
    rejected = 0

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

        # Detect ORB features in background regions only
        kp1, des1 = orb.detectAndCompute(prev_gray, bg_mask)
        kp2, des2 = orb.detectAndCompute(curr_gray, bg_mask)

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
                 f"rejected={rejected}, max_drift={max_translation_px:.0f}px)")
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


def outpaint_background(bg_image_path: str, target_width: int, target_height: int,
                        scale_factor: float = 1.3, cache_dir: str = "/tmp/bg_outpaint_cache") -> str:
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
    target_aspect = required_w / required_h  # 9:16 비율
    orig_aspect = orig_w / orig_h
    
    # Check if aspect ratio is already correct (within 2% tolerance)
    aspect_diff = abs(orig_aspect - target_aspect) / target_aspect
    
    # Case 1: Background is large enough (can downscale) → Crop to aspect ratio + resize
    if orig_w >= required_w and orig_h >= required_h:
        logging.info(f"Outpaint: Background {orig_w}x{orig_h} >= required {required_w}x{required_h}, cropping to aspect ratio")
        
        # Calculate crop dimensions to match target aspect ratio
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
        
        # Resize to required dimensions (downscale)
        resized = cv2.resize(cropped, (required_w, required_h), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(cache_path, resized)
        
        logging.info(f"Outpaint: Cropped {orig_w}x{orig_h} → {crop_w}x{crop_h} → resized to {required_w}x{required_h}")
        return cache_path
    
    # Case 2: Background is too small → Need outpainting (never upscale with resize)
    # First, ensure correct aspect ratio, then extend to required size
    
    if aspect_diff >= 0.02:
        # Aspect ratio is wrong → First crop to correct aspect ratio
        logging.info(f"Outpaint: Adjusting aspect ratio from {orig_aspect:.3f} to {target_aspect:.3f}")
        
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
        logging.info(f"Outpaint: Cropped to correct aspect ratio: {orig_w}x{orig_h}")
    
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
        
        logging.info(f"FLUX Outpaint: Loading FLUX.1 Fill [dev]...")
        
        # Load pipeline from Hugging Face
        pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            torch_dtype=torch.bfloat16,
            cache_dir="/mnt/models",
        ).to("cuda:0")
        
        orig_h, orig_w = bg_original.shape[:2]
        
        # Step 1: Create canvas with original image centered
        canvas_np = np.zeros((final_h, final_w, 3), dtype=np.uint8)
        canvas_np[margin_h:margin_h+orig_h, margin_w:margin_w+orig_w] = bg_original
        
        # Step 2: Create mask (white = areas to fill, black = keep original)
        mask_np = np.ones((final_h, final_w), dtype=np.uint8) * 255
        mask_np[margin_h:margin_h+orig_h, margin_w:margin_w+orig_w] = 0  # Keep original center
        
        # Convert to PIL
        canvas_pil = Image.fromarray(cv2.cvtColor(canvas_np, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask_np)
        
        # Step 3: Use FLUX.1 Fill to outpaint
        prompt = "seamless background extension, natural continuation, consistent lighting and style, photorealistic, high quality"
        
        logging.info(f"FLUX Outpaint: Generating outpainted image with FLUX.1 Fill...")
        
        generated = pipe(
            image=canvas_pil,
            mask_image=mask_pil,
            prompt=prompt,
            height=final_h,
            width=final_w,
            num_inference_steps=30,  # FLUX.1 Fill needs more steps than FLUX.2
            guidance_scale=30.0,  # High guidance for better adherence
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


# Placeholder for other functions that might be in the original file
def extract_camera_motion(*args, **kwargs):
    """Placeholder - implement actual camera motion extraction"""
    pass

def warp_background_video(*args, **kwargs):
    """Placeholder - implement actual background warping"""
    pass
def warp_background_video(bg_image_path: str, video_path: str,
                          output_path: str, force_rate: int = 16,
                          scale_factor: float = 1.3) -> str:
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

    # Step 2: Prepare background image with outpainting for better quality
    target_w, target_h = vid_w, vid_h
    needed_w = int(target_w * scale_factor)
    needed_h = int(target_h * scale_factor)
    
    # Use outpainting to expand background (preserves center quality)
    outpainted_bg_path = outpaint_background(bg_image_path, target_w, target_h, scale_factor)
    
    # Load the outpainted/processed background
    bg_img = cv2.imread(outpainted_bg_path)
    if bg_img is None:
        logging.warning(f"CameraMotion: Failed to read processed bg image")
        return ""
    
    bg_h, bg_w = bg_img.shape[:2]
    
    # If outpainting produced exact size, use it directly
    # Otherwise, resize to needed dimensions
    if bg_w != needed_w or bg_h != needed_h:
        bg_scaled = cv2.resize(bg_img, (needed_w, needed_h), interpolation=cv2.INTER_LANCZOS4)
        logging.info(f"CameraMotion: Resized outpainted bg {bg_w}x{bg_h} → {needed_w}x{needed_h}")
    else:
        bg_scaled = bg_img
        logging.info(f"CameraMotion: Using outpainted bg at {bg_w}x{bg_h}")

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

        # Apply: first offset to scaled image center, then apply camera motion
        # We apply INVERSE of the camera motion to the background,
        # so the background moves opposite to the camera (creating the illusion
        # that the camera is panning over the background)
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            H_inv = np.eye(3)

        warp_H = T_center @ H_inv

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
