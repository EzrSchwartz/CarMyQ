# import asyncio
# from http.server import BaseHTTPRequestHandler, HTTPServer
# import json
# from simplefunctions import writeJson, readJson
# from datetime import datetime
# import os
# from Segment import Segment
# HOST = "0.0.0.0"
# PORT = 8000

# # Create images directory if it doesn't exist
# os.makedirs("images", exist_ok=True)
# count = 0
# class RequestHandler(BaseHTTPRequestHandler):
#     def do_GET(self):
#         if self.path == "/command":
#             self.send_response(200)
#             self.end_headers()

#             command = readJson("command")
#             self.wfile.write(command.encode())

#             writeJson("command", "None")
#             print(f"Sent command: {command}")

#     def do_POST(self):
#         content_length = int(self.headers['Content-Length'])
#         content_type = self.headers.get('Content-Type', '')
        
#         post_data = self.rfile.read(content_length)

#         # Check if it's an image upload FIRST (before JSON parsing)
#         if content_type == 'image/jpeg' or self.path == "/upload":
#             try:
#                 # Save image with timestamp
#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 filename = f"images/image_{timestamp}.jpg"

#                 with open(filename, 'wb') as f:
#                     f.write(post_data)
                
#                 print(f"✓ Image saved: {filename} ({len(post_data)} bytes)")
                
#                 self.send_response(200)
#                 self.end_headers()
#                 self.wfile.write(b"Image received")

#                 Segment(filename)
#                 os.remove(filename)
#                 return
            
#             except Exception as e:
#                 print(f"✗ Error saving image: {e}")
#                 self.send_response(500)
#                 self.end_headers()
#                 self.wfile.write(b"Error")
#                 return

#         # Original JSON door state handling (only if not an image)
#         try:
#             data_received = json.loads(post_data)
#             door_state = data_received.get("door_state", "unknown")
#             print(f"Door state update: {door_state}")
#             writeJson("DoorState", door_state)
            
#             self.send_response(200)
#             self.end_headers()
#             self.wfile.write(b"OK")
#         except json.JSONDecodeError:
#             print("Invalid JSON received")
#             self.send_response(400)
#             self.end_headers()
#             self.wfile.write(b"Invalid JSON")


# def start_server():
#     server = HTTPServer((HOST, PORT), RequestHandler)
#     print(f"Server running on {HOST}:{PORT}")
#     server.serve_forever()


# async def run_server():
#     loop = asyncio.get_running_loop()
#     await loop.run_in_executor(None, start_server)


import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from simplefunctions import writeJson, readJson
from datetime import datetime
import os
import threading
import queue
import cv2
import numpy as np

from Segment import Segment

HOST = "0.0.0.0"
PORT = 8000

os.makedirs("images", exist_ok=True)

# Background segmentation queue
segment_queue = queue.Queue()

def segment_worker():
    while True:
        filename = segment_queue.get()
        try:
            Segment(filename)
        except Exception as e:
            print("✗ Segment error:", e)
        finally:
            try:
                os.remove(filename)
            except:
                pass
            segment_queue.task_done()

threading.Thread(target=segment_worker, daemon=True).start()


class RequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/command":
            self.send_response(200)
            self.end_headers()

            command = readJson("command")
            self.wfile.write(command.encode())
            writeJson("command", "None")

            print(f"Sent command: {command}")

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        content_type = self.headers.get("Content-Type", "")

        post_data = self.rfile.read(content_length)

        # ---------- IMAGE UPLOAD ----------
        if self.path == "/upload":

            if "image/jpeg" not in content_type:
                self.send_response(415)
                self.end_headers()
                self.wfile.write(b"Unsupported Media Type")
                return

            try:
                # Decode image safely
                np_arr = np.frombuffer(post_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    raise ValueError("Invalid JPEG data")

                # Save verified image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"images/image_{timestamp}.jpg"
                cv2.imwrite(filename, frame)

                print(f"✓ Image saved: {filename}")

                # Push to background worker
                segment_queue.put(filename)

                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")
                return

            except Exception as e:
                print("✗ Image upload error:", e)
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"Error")
                return

        # ---------- JSON DOOR STATE ----------
        try:
            data = json.loads(post_data)
            door_state = data.get("door_state", "unknown")

            writeJson("DoorState", door_state)
            print(f"Door state update: {door_state}")

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Invalid JSON")


def start_server():
    server = HTTPServer((HOST, PORT), RequestHandler)
    print(f"Server running on {HOST}:{PORT}")
    server.serve_forever()


async def run_server():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, start_server)
