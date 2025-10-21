#!/usr/bin/env python3
"""
Simple HTTP server for testing FastPlaid + mxbai-edge-colbert demo
Handles CORS headers needed for WASM and ES modules
"""

import http.server
import socketserver
import os
import sys
from urllib.parse import urlparse

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    
    # Change to demo directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    with socketserver.TCPServer(("", port), CORSHTTPRequestHandler) as httpd:
        print(f"🚀 FastPlaid + mxbai-edge-colbert demo server running at:")
        print(f"   http://localhost:{port}")
        print(f"   http://127.0.0.1:{port}")
        print()
        print("📝 Features:")
        print("   - CORS headers for WASM")
        print("   - ES modules support")
        print("   - pylate-rs integration")
        print("   - Real Hugging Face model loading")
        print()
        print("Press Ctrl+C to stop")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")

if __name__ == "__main__":
    main()