#!/usr/bin/env python3
"""
AI Biometric Fingerprint Recognition & Forensic Classification Application Launcher.
Starts FastAPI server on http://127.0.0.1:8000 with hot-reloading for local testing and demonstration.
"""

import os
import sys
import uvicorn

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("RELOAD", "true").lower() in ("true", "1", "yes")

    print("=" * 72)
    print(" AI BIOMETRIC FINGERPRINT FORENSIC INTELLIGENCE SYSTEM (STAGE 10)")
    print("=" * 72)
    print(f" Dashboard URL : http://{host}:{port}")
    print(f" API Docs URL  : http://{host}:{port}/docs")
    print(f" Health Check  : http://{host}:{port}/api/health")
    print(f" Reload Mode   : {reload}")
    print("=" * 72)

    uvicorn.run("src.app.main:app", host=host, port=port, reload=reload)
