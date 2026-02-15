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
    Smooth cumulative homographies to reduce jitter.

    Applies a moving average to the translation components (tx, ty)
    and rotation angle to produce smoother camera motion.
    """
    if len(homographies) <= 1:
        return homographies

    n = len(homographies)

    # Decompose into translation + rotation + scale for smoothing
    tx_arr = np.array([H[0, 2] for H in homographies])
    ty_arr = np.array([H[1, 2] for H in homographies])
    # Approximate rotation angle from H
    angle_arr = np.array([np.arctan2(H[1, 0], H[0, 0]) for H in homographies])
    scale_arr = np.array([np.sqrt(H[0, 0]**2 + H[1, 0]**2) for H in homographies])

    # Moving average filter
    half = window // 2
    tx_smooth = np.copy(tx_arr)
    ty_smooth = np.copy(ty_arr)
    angle_smooth = np.copy(angle_arr)
    scale_smooth = np.copy(scale_arr)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        tx_smooth[i] = np.mean(tx_arr[lo:hi])
        ty_smooth[i] = np.mean(ty_arr[lo:hi])
        angle_smooth[i] = np.mean(angle_arr[lo:hi])
        scale_smooth[i] = np.mean(scale_arr[lo:hi])

    # Reconstruct homographies from smoothed components
    smoothed = []
    for i in range(n):
        cos_a = np.cos(angle_smooth[i]) * scale_smooth[i]
        sin_a = np.sin(angle_smooth[i]) * scale_smooth[i]
        H = np.array([
            [cos_a, -sin_a, tx_smooth[i]],
            [sin_a,  cos_a, ty_smooth[i]],
            [0,      0,     1]
        ], dtype=np.float64)
        smoothed.append(H)

    return smoothed


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

    # Step 2: Prepare background image (scale up for panning room)
    bg_img = cv2.imread(bg_image_path)
    if bg_img is None:
        logging.warning(f"CameraMotion: Failed to read bg image")
        return ""

    # Target size matches video dimensions
    target_w, target_h = vid_w, vid_h

    # Scale up background to provide panning room
    scaled_w = int(target_w * scale_factor)
    scaled_h = int(target_h * scale_factor)
    bg_scaled = cv2.resize(bg_img, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)

    # Center offset (the initial view is the center crop of the scaled image)
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
