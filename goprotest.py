#pip install open-gopro opencv-python

import asyncio
import cv2
from open_gopro import WiredGoPro
from open_gopro.models.constants import Toggle

async def stream_to_computer():
    # Connect to GoPro via USB
    async with WiredGoPro() as gopro:
        print(f"Connected to GoPro via USB: {gopro.identifier}")

        # Start the preview stream
        await gopro.http_command.set_shutter(shutter=Toggle.DISABLE)
        await gopro.http_command.set_preview_stream(mode=Toggle.ENABLE)

        # Get the stream URL (for wired connection, use localhost)
        stream_url = "udp://127.0.0.1:8554"
        print(f"Stream URL: {stream_url}")

        # Open the stream with OpenCV
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            print("Error: Could not open video stream")
            return

        print("Streaming... Press 'q' to quit")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                cv2.imshow('GoPro Stream', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            await gopro.http_command.set_preview_stream(mode=Toggle.DISABLE)
            print("Stream stopped")

asyncio.run(stream_to_computer())
# Note: This code assumes you have the open_gopro library installed and a compatible GoPro camera.

