"""
run_app.py — Launcher for Gridlock Streamlit app
  Run this from the project root: python run_app.py
  It sets the correct Python path and launches streamlit.
"""
import subprocess
import sys
import os

# Ensure we're in the project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add project root to path
sys.path.insert(0, os.getcwd())

print("=" * 60)
print("  🚦 GRIDLOCK — Traffic Violation Detection")
print("=" * 60)
print(f"  Working dir : {os.getcwd()}")
print(f"  Python      : {sys.executable}")
print()

# Check key dependencies
missing = []
for pkg in ["ultralytics", "streamlit", "cv2", "pandas", "numpy"]:
    try:
        __import__(pkg)
        print(f"  ✅ {pkg}")
    except ImportError:
        print(f"  ❌ {pkg} — not installed")
        missing.append(pkg)

if missing:
    print(f"\n⚠ Missing packages: {', '.join(missing)}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

print()
print("  Launching Streamlit dashboard...")
print("  → http://localhost:8501")
print("=" * 60)

subprocess.run([
    sys.executable, "-m", "streamlit", "run", "src/app.py",
    "--server.port", "8501",
    "--server.headless", "false",
    "--browser.gatherUsageStats", "false",
])
