
# import asyncio
# from http.server import BaseHTTPRequestHandler, HTTPServer
# import json
# from simplefunctions import writeJson, readJson

# HOST = "0.0.0.0"
# PORT = 8000

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
#         post_data = self.rfile.read(content_length)

#         try:
#             data_received = json.loads(post_data)
#             door_state = data_received.get("door_state", "unknown")
#             print(f"Door state update: {door_state}")
#             writeJson("DoorState", door_state)
#         except json.JSONDecodeError:
#             print("Invalid JSON received")

#         self.send_response(200)
#         self.end_headers()
#         self.wfile.write(b"OK")


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

HOST = "0.0.0.0"
PORT = 8000

# Create images directory if it doesn't exist
os.makedirs("images", exist_ok=True)

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
        content_length = int(self.headers['Content-Length'])
        content_type = self.headers.get('Content-Type', '')
        
        post_data = self.rfile.read(content_length)

        # Check if it's an image upload FIRST (before JSON parsing)
        if content_type == 'image/jpeg' or self.path == "/upload":
            try:
                # Save image with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"images/image_{timestamp}.jpg"
                
                with open(filename, 'wb') as f:
                    f.write(post_data)
                
                print(f"✓ Image saved: {filename} ({len(post_data)} bytes)")
                
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"Image received")
                return
            
            except Exception as e:
                print(f"✗ Error saving image: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"Error")
                return

        # Original JSON door state handling (only if not an image)
        try:
            data_received = json.loads(post_data)
            door_state = data_received.get("door_state", "unknown")
            print(f"Door state update: {door_state}")
            writeJson("DoorState", door_state)
            
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        except json.JSONDecodeError:
            print("Invalid JSON received")
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