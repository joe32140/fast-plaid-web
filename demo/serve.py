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
        # More permissive CORS headers for WASM and model loading
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, HEAD')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Access-Control-Expose-Headers', '*')
        
        # Less restrictive COEP/COOP for development
        # Comment out the strict policies that are blocking Hugging Face
        # self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        # self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        
        # Alternative: Use credentialless for WASM compatibility
        self.send_header('Cross-Origin-Embedder-Policy', 'credentialless')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    
    # Change to demo directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if --simple flag is provided for basic CORS
    simple_mode = '--simple' in sys.argv
    
    if simple_mode:
        print("🔧 Running in simple mode (basic CORS, no COEP)")
        
        class SimpleCORSHandler(http.server.SimpleHTTPRequestHandler):
            def end_headers(self):
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, HEAD')
                self.send_header('Access-Control-Allow-Headers', '*')
                super().end_headers()
            
            def do_OPTIONS(self):
                self.send_response(200)
                self.end_headers()
        
        handler_class = SimpleCORSHandler
    else:
        handler_class = CORSHTTPRequestHandler
    
    with socketserver.TCPServer(("", port), handler_class) as httpd:
        print(f"🚀 FastPlaid + mxbai-edge-colbert demo server running at:")
        print(f"   http://localhost:{port}")
        print(f"   http://127.0.0.1:{port}")
        print()
        print("📝 Features:")
        print("   - CORS headers for WASM")
        print("   - ES modules support") 
        print("   - pylate-rs integration")
        print("   - Real Hugging Face model loading")
        if simple_mode:
            print("   - Simple CORS mode (no COEP restrictions)")
        else:
            print("   - Full COEP/COOP headers")
        print()
        print("💡 If you get CORS errors, try: python3 serve.py --simple")
        print("Press Ctrl+C to stop")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")

if __name__ == "__main__":
    main()