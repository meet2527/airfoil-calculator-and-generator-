import webview
import threading
import uvicorn
import time
import socket
import os
import sys
from api import app

def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def start_server():
    """Start the FastAPI server."""
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

if __name__ == "__main__":
    # 1. Start FastAPI in a background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # 2. Wait for the server to be ready
    print("Starting background server...")
    max_retries = 10
    while not is_port_open(8000) and max_retries > 0:
        time.sleep(1)
        max_retries -= 1

    if max_retries == 0:
        print("Error: Server failed to start.")
        exit(1)

    print("Server ready. Launching window...")

    # 3. Create and start the webview window
    # We use a nice 'Aero' looking title and a large default size
    window = webview.create_window(
        'RC Airfoil Configuration System',
        'http://127.0.0.1:8000',
        width=1400,
        height=900,
        min_size=(1000, 700),
        confirm_close=True
    )

    webview.start()
