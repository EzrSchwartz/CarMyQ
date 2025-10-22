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
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

from PIL import Image
from resnetclassification import classify

# ==============================
# CONFIGURATION
# ==============================

extractor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
CAR_CLASS_ID = 13  # for Cityscapes, class 13 = car

# Choose mode: "livestream" or "video"
MODE = "video"  # Change to "livestream" for GoPro streaming
VIDEO_PATH = r'c:\Users\ezran\OneDrive\Desktop\GX020089.MP4' # Path to your MP4 file

SAVE_DIR = "D:\VehiclesData\Other"
IMG_SIZE = (256, 256)  # must match your model's input size
SHOW_OVERLAY = True    # show real-time overlay
SAVE_INTERVAL = 1     # save every Nth frame to limit data size

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
                                # Convert logits to probabilities (softmax along class dimension)
                probs = torch.nn.functional.softmax(upsampled_logits, dim=1)

                # Extract the probability map for the "car" class
                car_probs = probs[0, CAR_CLASS_ID, :, :].cpu().numpy()

                # Apply your confidence threshold (e.g., 0.8 = 80%)
                CONF_THRESHOLD = 0.8
                car_mask = (car_probs > CONF_THRESHOLD).astype(np.uint8)
                # full_mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

            # --- Filter for car class only ---
            # Create binary mask: 1 where car is detected, 0 otherwise
            # car_mask = (full_mask == CAR_CLASS_ID).astype(np.uint8)

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
                # Compute percent of car pixels in frame
                car_pixel_ratio = car_mask.sum() / (car_mask.shape[0] * car_mask.shape[1])

                # Trigger only if above threshold
                THRESHOLD = 0.0 # 5% of frame area (adjust this as needed)

                if car_pixel_ratio > THRESHOLD:
                    print(f"🚗 Car detected ({car_pixel_ratio*100:.2f}% of frame) — processing...")
                    frame_path = f"{SAVE_DIR}/frames/frame_{frame_idx:05d}(night)(GX020089).png"
                    cv2.imwrite(frame_path, frame)
                    cv2.imwrite(f"{SAVE_DIR}/labels/mask_{frame_idx:05d}(night)(GX020089).png", mask_resized)


                    # classify(frame)
                else:
                    print(f"⏭️  Skipped frame {frame_idx}: only {car_pixel_ratio*100:.2f}% car coverage")
                # if car_mask.sum() > 0:
                #     frame_path = f"{SAVE_DIR}/frames/frame_{frame_idx:05d}.png"
                #     mask_path = f"{SAVE_DIR}/labels/mask_{frame_idx:05d}.png"
                #     cv2.imwrite(frame_path, frame)
                #     # Save the binary car mask (0 or 255)
                #     # cv2.imwrite(mask_path, car_mask * 255)
                #     print(f"💾 Saved frame {frame_idx}/{total_frames}: {frame_path} (car detected: {car_mask.sum()} pixels)")
                # else:
                #     print(f"⏭️  Skipped frame {frame_idx}/{total_frames}: No car detected")

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


