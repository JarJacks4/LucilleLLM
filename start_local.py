#!/usr/bin/env python3
"""
Start LucilleLLM locally for testing
"""

import os
import sys
import subprocess

def start_server():
    """Start the local development server"""
    print("🚀 Starting LucilleLLM local server...")
    print("📍 Server will be available at: http://localhost:8080")
    print("📝 API Documentation: http://localhost:8080/docs")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Check if .env exists
    if not os.path.exists('.env'):
        print("⚠️  Warning: No .env file found!")
        print("💡 The app will work but without OpenAI functionality")
        print("💡 To enable full functionality:")
        print("   1. Get API key: https://platform.openai.com/api-keys")
        print("   2. Create .env: echo 'OPENAI_API_KEY=sk-your-key' > .env")
        print()
    
    try:
        # Start uvicorn with hot reload
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8080", 
            "--reload",
            "--log-level", "info"
        ]
        
        subprocess.run(cmd, env=env, cwd=os.getcwd())
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

if __name__ == "__main__":
    start_server()

