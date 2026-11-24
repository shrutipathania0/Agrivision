"""
AgriVision Single-Command Application Launcher
Starts both the FastAPI Backend (port 8000) and React Vite Frontend (port 5173).
"""

import sys
import time
import subprocess
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
PYTHON_EXE = ROOT_DIR / ".venv" / "Scripts" / "python.exe"


def main():
    print("=" * 70)
    print("AGRIVISION APPLICATION LAUNCHER")
    print("Dual-AI Crop Disease Detection & Agricultural Advisory System")
    print("=" * 70)

    # 1. Start FastAPI Backend
    print("\n[1/2] Starting FastAPI Backend on http://127.0.0.1:8000...")
    backend_cmd = [
        str(PYTHON_EXE), "-m", "uvicorn", "backend.app.main:app",
        "--host", "127.0.0.1", "--port", "8000"
    ]
    backend_proc = subprocess.Popen(backend_cmd, cwd=str(ROOT_DIR))

    # Give backend a moment to initialize CUDA tensors
    time.sleep(2)

    # 2. Start Vite Frontend
    print("[2/2] Starting React Vite Frontend on http://localhost:5173...")
    frontend_cmd = ["npm.cmd", "run", "dev"]
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=str(FRONTEND_DIR), shell=True)

    print("\n" + "=" * 70)
    print("AGRIVISION IS LIVE AND OPERATIONAL!")
    print("  -> Web Dashboard : http://localhost:5173")
    print("  -> REST API Docs : http://127.0.0.1:8000/docs")
    print("  -> Health Check  : http://127.0.0.1:8000/api/health")
    print("Press Ctrl+C to terminate both servers.")
    print("=" * 70 + "\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down AgriVision servers...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
