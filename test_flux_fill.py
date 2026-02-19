#!/usr/bin/env python3
"""
Test FLUX.1 Fill outpainting
"""
import sys
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import the outpainting function
from camera_motion import outpaint_background

def test_flux_fill_outpaint():
    """Test FLUX.1 Fill outpainting with a sample image"""
    
    # Load test background image
    bg_path = "/home/ubuntu/WanAvatar/background/stages/upload_85e2149e-6cbc-43c1-b6f4-c8c2a248cb9b.png"
    
    # Target size for Dance Shorts (9:16 portrait)
    target_w = 720
    target_h = 1280
    scale_factor = 1.3  # 30% expansion
    
    print(f"Testing with: {bg_path}")
    print(f"Target size: {target_w}x{target_h}")
    print(f"Scale factor: {scale_factor} (30% margin for camera motion)")
    print(f"Required size: {int(target_w * scale_factor)}x{int(target_h * scale_factor)}")
    
    # Run outpaint_background (handles crop/resize/outpaint logic)
    print("\nRunning outpaint_background...")
    result_path = outpaint_background(
        bg_image_path=bg_path,
        target_width=target_w,
        target_height=target_h,
        scale_factor=scale_factor
    )
    
    if result_path and result_path != bg_path:
        print(f"\n✅ Success! Result saved to: {result_path}")
        
        # Check result size
        result_img = cv2.imread(result_path)
        if result_img is not None:
            res_h, res_w = result_img.shape[:2]
            print(f"Result size: {res_w}x{res_h}")
            
            # Copy to uploads for easy viewing
            import shutil
            output_path = "/home/ubuntu/WanAvatar/uploads/flux_fill_85e2149e_result.png"
            shutil.copy(result_path, output_path)
            print(f"Copied to: {output_path}")
        else:
            print("Warning: Could not read result image")
    else:
        print(f"\n✅ No processing needed, using original: {result_path}")

if __name__ == "__main__":
    test_flux_fill_outpaint()
