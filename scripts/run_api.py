"""
CLI Script to launch the FastAPI server.
Usage:
    python scripts/run_api.py
"""
import sys
from pathlib import Path
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import settings

if __name__ == "__main__":
    uvicorn.run("api.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
