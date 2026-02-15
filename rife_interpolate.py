"""
RIFE (Real-Time Intermediate Flow Estimation) frame interpolation helper.
Uses Practical-RIFE v4.25 model for AI-based frame interpolation.
"""
import os
import sys
import cv2
import torch
import numpy as np
import logging
from torch.nn import functional as F

# Add rife_model to path
RIFE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rife_model")

_rife_model = None
_rife_device = None


def _load_rife_model():
    """Lazy-load RIFE model (singleton)."""
    global _rife_model, _rife_device
    if _rife_model is not None:
        return _rife_model

    sys.path.insert(0, RIFE_DIR)
    from train_log.RIFE_HDv3 import Model

    _rife_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _rife_model = Model()
    _rife_model.load_model(os.path.join(RIFE_DIR, "train_log"), -1)
    _rife_model.eval()
    _rife_model.device()
    logging.info(f"RIFE v4.25 model loaded on {_rife_device}")
    return _rife_model


def interpolate_video_rife(input_path: str, target_fps: int = 30, scale: float = 1.0) -> str:
    """
    Interpolate video frames using RIFE AI model.

    Args:
        input_path: Path to input video file.
        target_fps: Target output FPS. Frames will be interpolated to reach this.
        scale: Processing scale (0.5 for high-res videos, 1.0 for normal).

    Returns:
        Path to the interpolated video (replaces input in-place).
    """
    model = _load_rife_model()
    device = _rife_device

    # Read video info
    cap = cv2.VideoCapture(input_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if orig_fps <= 0 or total_frames <= 1:
        cap.release()
        logging.warning(f"RIFE: Invalid video (fps={orig_fps}, frames={total_frames})")
        return input_path

    # Calculate interpolation multiplier
    multi = max(1, round(target_fps / orig_fps))
    if multi <= 1:
        cap.release()
        logging.info(f"RIFE: No interpolation needed (orig_fps={orig_fps:.1f} >= target_fps={target_fps})")
        return input_path

    out_fps = orig_fps * multi
    logging.info(f"RIFE: {orig_fps:.1f}fps -> {out_fps:.1f}fps ({multi}x), {total_frames} frames, {w}x{h}")

    # Padding for RIFE (must be divisible by 128/scale)
    tmp = max(128, int(128 / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)

    def pad_image(img):
        return F.pad(img, padding)

    # Output video
    output_path = input_path.replace(".mp4", "_rife.mp4")
    if output_path == input_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_rife{ext}"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (w, h))

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        cap.release()
        writer.release()
        return input_path

    with torch.no_grad():
        I1 = torch.from_numpy(np.transpose(frame[:, :, ::-1].copy(), (2, 0, 1))).to(device).unsqueeze(0).float() / 255.
        I1 = pad_image(I1)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            I0 = I1
            I1 = torch.from_numpy(np.transpose(frame[:, :, ::-1].copy(), (2, 0, 1))).to(device).unsqueeze(0).float() / 255.
            I1 = pad_image(I1)

            # Write original frame
            orig_frame = (I0[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
            writer.write(orig_frame[:, :, ::-1])

            # Generate intermediate frames
            for i in range(multi - 1):
                timestep = (i + 1) / multi
                if hasattr(model, 'version') and model.version >= 3.9:
                    mid = model.inference(I0, I1, timestep, scale)
                else:
                    mid = model.inference(I0, I1, scale)
                mid_frame = (mid[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
                writer.write(mid_frame[:, :, ::-1])

            frame_idx += 1
            if frame_idx % 50 == 0:
                logging.info(f"RIFE: Processed {frame_idx}/{total_frames} frames")

        # Write last frame
        last_frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
        writer.write(last_frame[:, :, ::-1])

    cap.release()
    writer.release()

    # Re-encode to H.264 for browser compatibility (mp4v isn't always supported)
    h264_path = output_path.replace("_rife.mp4", "_rife_h264.mp4")
    try:
        import subprocess
        result = subprocess.run([
            "ffmpeg", "-y", "-i", output_path,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-an",  # no audio (will be merged separately)
            h264_path,
        ], capture_output=True, timeout=300)
        if result.returncode == 0 and os.path.exists(h264_path) and os.path.getsize(h264_path) > 0:
            os.replace(h264_path, input_path)
            os.remove(output_path)
            logging.info(f"RIFE: Interpolation complete, saved to {input_path}")
        else:
            logging.warning(f"RIFE: H.264 re-encode failed, using mp4v output")
            os.replace(output_path, input_path)
    except Exception as e:
        logging.warning(f"RIFE: H.264 re-encode error: {e}")
        os.replace(output_path, input_path)

    return input_path
