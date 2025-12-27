
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

IMG_SIZE = (256, 256)  # must match your model's input size
SHOW_OVERLAY = True    # show real-time overlay
SAVE_INTERVAL = 1     # save every Nth frame to limit data size

# ==============================
# SETUP
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor()
])




async def StreamSegment():
    """Process GoPro livestream"""
    try:
        # Try common GoPro USB IP addresses
        # GoPro cameras create a network interface when connected via USB
        gopro_ips = ["172.20.110.51", "172.21.110.51", "172.22.110.51", "172.23.110.51", "172.24.110.51", "172.25.110.51"]

        gopro = None
        for ip in gopro_ips:
            try:
                print(f"Trying to connect to GoPro at {ip}...")
                gopro = WiredGoPro(target=ip, enable_wifi=False)
                await gopro.open()
                print(f"Connected to GoPro via USB: {gopro.identifier}")
                break
            except Exception as e:
                print(f"  Failed: {e}")
                continue




        if gopro is None:
            raise Exception("Could not find GoPro on any USB network interface")

        await gopro.http_command.set_shutter(shutter=Toggle.DISABLE)
        await gopro.http_command.set_preview_stream(mode=Toggle.ENABLE)

        stream_url = "udp://127.0.0.1:8554"
        print(f"Stream URL: {stream_url}")

        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("Could not open video stream")
            await gopro.close()
            return

        print("Streaming and segmenting... Press 'q' to quit.")
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
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

                frame_idx += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            await gopro.http_command.set_preview_stream(mode=Toggle.DISABLE)
            print("Stream stopped and model closed.")

    except Exception as e:
        print(f"Error connecting to GoPro: {e}")
        print("Make sure your GoPro is:")
        print("  1. Connected via USB-C cable")
        print("  2. Powered on")
        print("  3. In USB mode (not charging mode)")
    finally:
        try:
            await gopro.close()
        except:
            pass
