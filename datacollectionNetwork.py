#pip install open-gopro opencv-python torch torchvision

import asyncio
import cv2
import os
import torch
import numpy as np
from torchvision import transforms
from open_gopro import WiredGoPro
from open_gopro.models.constants import Toggle
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
from resnetclassification import classify

# ==============================
# CONFIGURATION
# ==============================
extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# Car class ID in ADE20K dataset (class 21 is 'car')
CAR_CLASS_ID = 21

# Choose mode: "livestream" or "video"
MODE = "video"  # Change to "livestream" for GoPro streaming
VIDEO_PATH = "path/to/your/video.mp4"  # Path to your MP4 file

SAVE_DIR = "segmented_gopro_dataset"
IMG_SIZE = (256, 256)  # must match your model's input size
SHOW_OVERLAY = True    # show real-time overlay
SAVE_INTERVAL = 5      # save every Nth frame to limit data size

# ==============================
# SETUP
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"🖥️  Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor()
])

os.makedirs(f"{SAVE_DIR}/frames", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/labels", exist_ok=True)

# ==============================
# STREAM + SEGMENT
# ==============================
async def stream_and_segment_livestream():
    """Process GoPro livestream"""
    async with WiredGoPro() as gopro:
        print(f"✅ Connected to GoPro via USB: {gopro.identifier}")

        await gopro.http_command.set_shutter(shutter=Toggle.DISABLE)
        await gopro.http_command.set_preview_stream(mode=Toggle.ENABLE)

        stream_url = "udp://127.0.0.1:8554"
        print(f"🎥 Stream URL: {stream_url}")

        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("❌ Could not open video stream")
            return

        print("🚗 Streaming and segmenting... Press 'q' to quit.")
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Failed to grab frame")
                    break

                # --- Preprocess for Da ---
                # Convert BGR to RGB (OpenCV uses BGR, model expects RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                with torch.no_grad():
                    # Extract features and move to device
                    inputs = extractor(images=frame_rgb, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(device)

                    # Run inference
                    outputs = model(pixel_values=pixel_values)
                    logits = outputs.logits  # (batch, num_classes, height, width)

                    # Upsample logits to original image size
                    upsampled_logits = torch.nn.functional.interpolate(
                        logits,
                        size=(frame.shape[0], frame.shape[1]),
                        mode="bilinear",
                        align_corners=False
                    )

                    # Get predicted class for each pixel
                    full_mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

                # --- Filter for car class only ---
                # Create binary mask: 1 where car is detected, 0 otherwise
                car_mask = (full_mask == CAR_CLASS_ID).astype(np.uint8)

                # --- Convert mask to uint8 for visualization ---
                # Scale to 0-255 for better visibility
                mask_resized = car_mask * 255

                # --- Optional overlay ---
                if SHOW_OVERLAY:
                    overlay = cv2.addWeighted(
                        frame, 0.7,
                        cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR), 0.3, 0
                    )
                    cv2.imshow('Segmented Stream', overlay)
                else:
                    cv2.imshow('Segmented Stream', frame)

                # --- Save every Nth frame (only if car is detected) ---
                if frame_idx % SAVE_INTERVAL == 0:
                    # Only save if car is detected in the frame
                    if car_mask.sum() > 0:
                        classify(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            await gopro.http_command.set_preview_stream(mode=Toggle.DISABLE)
            print("🛑 Stream stopped and model closed.")


def process_video_file():
    """Process video file from disk"""
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video file not found: {VIDEO_PATH}")
        return

    print(f"🎥 Opening video file: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ Could not open video file")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📹 Video info: {total_frames} frames at {fps:.2f} FPS")
    print("🚗 Processing video... Press 'q' to quit.")

    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✅ Finished processing video")
                break

            # --- Preprocess for Da ---
            # Convert BGR to RGB (OpenCV uses BGR, model expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with torch.no_grad():
                # Extract features and move to device
                inputs = extractor(images=frame_rgb, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)

                # Run inference
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits  # (batch, num_classes, height, width)

                # Upsample logits to original image size
                upsampled_logits = torch.nn.functional.interpolate(
                    logits,
                    size=(frame.shape[0], frame.shape[1]),
                    mode="bilinear",
                    align_corners=False
                )

                # Get predicted class for each pixel
                full_mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

            # --- Filter for car class only ---
            # Create binary mask: 1 where car is detected, 0 otherwise
            car_mask = (full_mask == CAR_CLASS_ID).astype(np.uint8)

            # --- Convert mask to uint8 for visualization ---
            # Scale to 0-255 for better visibility
            mask_resized = car_mask * 255

            # --- Optional overlay ---
            if SHOW_OVERLAY:
                overlay = cv2.addWeighted(
                    frame, 0.7,
                    cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR), 0.3, 0
                )
                cv2.imshow('Segmented Video', overlay)
            else:
                cv2.imshow('Segmented Video', frame)

            # --- Save every Nth frame (only if car is detected) ---
            if frame_idx % SAVE_INTERVAL == 0:
                # Only save if car is detected in the frame
                if car_mask.sum() > 0:
                    frame_path = f"{SAVE_DIR}/frames/frame_{frame_idx:05d}.png"
                    mask_path = f"{SAVE_DIR}/labels/mask_{frame_idx:05d}.png"
                    cv2.imwrite(frame_path, frame)
                    # Save the binary car mask (0 or 255)
                    cv2.imwrite(mask_path, car_mask * 255)
                    print(f"💾 Saved frame {frame_idx}/{total_frames}: {frame_path} (car detected: {car_mask.sum()} pixels)")
                else:
                    print(f"⏭️  Skipped frame {frame_idx}/{total_frames}: No car detected")

            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🛑 Video processing stopped.")


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    if MODE == "video":
        process_video_file()
    elif MODE == "livestream":
        asyncio.run(stream_and_segment_livestream())
    else:
        print(f"❌ Invalid MODE: {MODE}. Use 'video' or 'livestream'")


