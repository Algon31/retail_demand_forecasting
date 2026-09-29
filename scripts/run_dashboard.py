"""
CLI Script to launch the Streamlit dashboard.
Usage:
    python scripts/run_dashboard.py
"""
import sys
import subprocess
from pathlib import Path

dashboard_path = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
