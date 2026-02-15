"""
AMT-G (All-Pairs Multi-Field Transforms - Giant) frame interpolation helper.
Uses AMT-G model with dual-resolution processing for highest quality.
Supports recursive 2x interpolation for smoother fast motion.
"""
import os
import sys
import cv2
import torch
import numpy as np
import logging
import subprocess
from torch.nn import functional as F

# Add amt_model to path
AMT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "amt_model")
AMT_WEIGHTS_G = os.path.join(AMT_DIR, "amt-g.pth")
AMT_WEIGHTS_L = os.path.join(AMT_DIR, "amt-l.pth")

_amt_model = None
_amt_device = None
_amt_model_name = None


class InputPadder:
    """Pads images such that dimensions are divisible by divisor."""
    def __init__(self, dims, divisor=16):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2,
                     pad_ht // 2, pad_ht - pad_ht // 2]

    def pad(self, *inputs):
        if len(inputs) == 1:
            return F.pad(inputs[0], self._pad, mode='replicate')
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def _load_amt_model():
    """Lazy-load AMT-G model (singleton), fallback to AMT-L."""
    global _amt_model, _amt_device, _amt_model_name
    if _amt_model is not None:
        return _amt_model

    sys.path.insert(0, AMT_DIR)
    _amt_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try AMT-G first, fallback to AMT-L
    if os.path.exists(AMT_WEIGHTS_G):
        from networks.AMT_G import Model as AMT_G
        _amt_model = AMT_G(corr_radius=3, corr_lvls=4, num_flows=5)
        ckpt = torch.load(AMT_WEIGHTS_G, map_location="cpu", weights_only=False)
        if "state_dict" in ckpt:
            _amt_model.load_state_dict(ckpt["state_dict"])
        else:
            _amt_model.load_state_dict(ckpt)
        _amt_model_name = "AMT-G"
    elif os.path.exists(AMT_WEIGHTS_L):
        from networks.AMT_L import Model as AMT_L
        _amt_model = AMT_L(corr_radius=3, corr_lvls=4, num_flows=5)
        ckpt = torch.load(AMT_WEIGHTS_L, map_location="cpu", weights_only=False)
        if "state_dict" in ckpt:
            _amt_model.load_state_dict(ckpt["state_dict"])
        else:
            _amt_model.load_state_dict(ckpt)
        _amt_model_name = "AMT-L"
    else:
        raise FileNotFoundError(
            f"AMT weights not found. Run setup_amt.sh to download."
        )

    _amt_model.eval().to(_amt_device)
    logging.info(f"{_amt_model_name} model loaded on {_amt_device}")
    return _amt_model


def _interpolate_2x(input_path: str, output_path: str, pass_label: str = "") -> str:
    """
    Single pass of 2x frame interpolation using AMT model.
    Reads input_path, writes 2x interpolated video to output_path.
    """
    model = _load_amt_model()
    device = _amt_device
    label = _amt_model_name or "AMT"
    if pass_label:
        label = f"{label} {pass_label}"

    cap = cv2.VideoCapture(input_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if orig_fps <= 0 or total_frames <= 1:
        cap.release()
        logging.warning(f"{label}: Invalid video (fps={orig_fps}, frames={total_frames})")
        return input_path

    out_fps = orig_fps * 2
    logging.info(f"{label}: {orig_fps:.1f}fps -> {out_fps:.1f}fps (2x), {total_frames} frames, {w}x{h}")

    padder = InputPadder((h, w), divisor=16)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (w, h))

    ret, frame = cap.read()
    if not ret:
        cap.release()
        writer.release()
        return input_path

    with torch.no_grad():
        I1 = torch.from_numpy(
            frame[:, :, ::-1].copy().transpose(2, 0, 1)
        ).unsqueeze(0).float().to(device) / 255.0

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            I0 = I1
            I1 = torch.from_numpy(
                frame[:, :, ::-1].copy().transpose(2, 0, 1)
            ).unsqueeze(0).float().to(device) / 255.0

            # Write original frame
            orig_frame = (I0[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
            writer.write(orig_frame[:, :, ::-1])

            # Pad and generate mid frame (timestep=0.5)
            I0_pad, I1_pad = padder.pad(I0, I1)
            embt = torch.tensor(0.5).float().view(1, 1, 1, 1).to(device)
            result = model(I0_pad, I1_pad, embt, eval=True)
            mid = padder.unpad(result['imgt_pred'])
            mid_frame = (mid[0].clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
            writer.write(mid_frame[:, :, ::-1])

            frame_idx += 1
            if frame_idx % 20 == 0:
                logging.info(f"{label}: Processed {frame_idx}/{total_frames} frames")

        # Write last frame
        last_frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
        writer.write(last_frame[:, :, ::-1])

    cap.release()
    writer.release()
    logging.info(f"{label}: 2x pass complete ({total_frames} -> ~{total_frames * 2} frames)")
    return output_path


def interpolate_video_amt(input_path: str, target_fps: int = 48) -> str:
    """
    Interpolate video frames using AMT-G with recursive 2x interpolation.

    Recursive 2x is higher quality than direct Nx because each pass only
    needs to estimate motion for half the time interval (timestep=0.5),
    which is the model's sweet spot.

    16fps -> 32fps (pass 1) -> 64fps (pass 2) for target_fps=48

    Args:
        input_path: Path to input video file.
        target_fps: Target output FPS.

    Returns:
        Path to the interpolated video (replaces input in-place).
    """
    model = _load_amt_model()
    label = _amt_model_name or "AMT"

    cap = cv2.VideoCapture(input_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if orig_fps <= 0 or total_frames <= 1:
        logging.warning(f"{label}: Invalid video (fps={orig_fps}, frames={total_frames})")
        return input_path

    if orig_fps >= target_fps:
        logging.info(f"{label}: No interpolation needed (orig_fps={orig_fps:.1f} >= target_fps={target_fps})")
        return input_path

    # Calculate number of recursive 2x passes needed
    num_passes = 0
    current_fps = orig_fps
    while current_fps < target_fps:
        current_fps *= 2
        num_passes += 1

    final_fps = orig_fps * (2 ** num_passes)
    logging.info(f"{label}: Recursive {num_passes}x2 interpolation: "
                 f"{orig_fps:.1f}fps -> {final_fps:.1f}fps ({2**num_passes}x), "
                 f"{total_frames} frames")

    # Perform recursive 2x passes
    current_input = input_path
    temp_files = []

    for p in range(num_passes):
        pass_label = f"[pass {p+1}/{num_passes}]"
        base, ext = os.path.splitext(input_path)
        temp_output = f"{base}_amt_pass{p+1}{ext}"
        temp_files.append(temp_output)

        _interpolate_2x(current_input, temp_output, pass_label)

        # Clean up previous pass temp file (not the original input)
        if current_input != input_path and os.path.exists(current_input):
            os.remove(current_input)

        current_input = temp_output

    # Re-encode final result to H.264 for browser compatibility
    h264_path = input_path.replace(".mp4", "_amt_h264.mp4")
    if h264_path == input_path:
        base, ext = os.path.splitext(input_path)
        h264_path = f"{base}_amt_h264{ext}"

    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", current_input,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-an",
            h264_path,
        ], capture_output=True, timeout=600)
        if result.returncode == 0 and os.path.exists(h264_path) and os.path.getsize(h264_path) > 0:
            os.replace(h264_path, input_path)
            logging.info(f"{label}: Recursive interpolation complete, "
                         f"{orig_fps:.1f}fps -> {final_fps:.1f}fps, saved to {input_path}")
        else:
            logging.warning(f"{label}: H.264 re-encode failed, using mp4v output")
            os.replace(current_input, input_path)
            current_input = None
    except Exception as e:
        logging.warning(f"{label}: H.264 re-encode error: {e}")
        os.replace(current_input, input_path)
        current_input = None

    # Clean up temp files
    for tf in temp_files:
        if tf and os.path.exists(tf):
            try:
                os.remove(tf)
            except OSError:
                pass
    if h264_path != input_path and os.path.exists(h264_path):
        try:
            os.remove(h264_path)
        except OSError:
            pass

    return input_path
