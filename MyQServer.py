# # 


# from http.server import BaseHTTPRequestHandler, HTTPServer
# import json
# from simplefunctions import writeJson,readJson
# import asyncio
# HOST = "0.0.0.0"
# PORT = 8000
# command = "NONE" # Global variable

# class RequestHandler(BaseHTTPRequestHandler):
#     def do_GET(self):
#         global command  # Tell Python to use the global variable defined above
#         if self.path == "/command":
#             self.send_response(200)
#             self.end_headers()
            
#             # Read current command from JSON

#             command = readJson("command")
#             self.wfile.write(command.encode())

#             # Reset command to NONE after sending
#             writeJson("command","None")
#             print(f"Sent command: {command}")

#     def do_POST(self):
#         content_length = int(self.headers['Content-Length'])
#         post_data = self.rfile.read(content_length)
#         try:
#             data_received = json.loads(post_data)
#             door_state = data_received.get("door_state", "unknown")
#             print(f"Door state update: {door_state}")
#             writeJson("DoorState",door_state)
#             # Load existing file to update specific keys
#             # with open('serverupdates.json', 'r') as f:
#             #     current_data = json.load(f)
            
#             # current_data["DoorState"] = door_state
            
#             # # FIXED INDENTATION BELOW
#             # with open("serverupdates.json", 'w') as f:
#             #     json.dump(current_data, f, indent=4) # Now properly indented
                
#         except json.JSONDecodeError:
#             print("Invalid JSON received")
        
#         self.send_response(200)
#         self.end_headers()
#         self.wfile.write(b"OK")

# async def run_server():
#     server = HTTPServer((HOST, PORT), RequestHandler)
#     print(f"Server running on {HOST}:{PORT}")
#     server.serve_forever()


import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from simplefunctions import writeJson, readJson

HOST = "0.0.0.0"
PORT = 8000

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
        post_data = self.rfile.read(content_length)

        try:
            data_received = json.loads(post_data)
            door_state = data_received.get("door_state", "unknown")
            print(f"Door state update: {door_state}")
            writeJson("DoorState", door_state)
        except json.JSONDecodeError:
            print("Invalid JSON received")

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")


def start_server():
    server = HTTPServer((HOST, PORT), RequestHandler)
    print(f"Server running on {HOST}:{PORT}")
    server.serve_forever()


async def run_server():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, start_server)
