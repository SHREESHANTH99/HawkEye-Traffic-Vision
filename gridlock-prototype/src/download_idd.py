"""
download_idd.py — Download IDD segmentation dataset from Kaggle via kagglehub
  Run once to pull raw dataset into data/raw/

Prerequisites:
  - pip install kagglehub
  - Place kaggle.json at C:\\Users\\<YOU>\\.kaggle\\kaggle.json
    (Get from kaggle.com → Account → API → Create New Token)
"""

import os
import sys
import shutil
from pathlib import Path

try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
except ImportError:
    print("❌ kagglehub not installed. Run: pip install kagglehub[pandas-datasets]")
    sys.exit(1)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Gridlock — IDD Dataset Download")
print("=" * 60)

# ─── Option 1: Load as Pandas DataFrame (metadata / CSV inspection) ───────────
print("\n[1/2] Loading IDD dataset metadata via KaggleDatasetAdapter...")
try:
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "sovitrath/indian-driving-dataset-segmentation-part-2",
        "",
    )
    print(f"      Rows loaded : {len(df)}")
    print(f"      Columns     : {list(df.columns)}")
    print(f"      Sample:\n{df.head(3).to_string()}\n")
    df.to_csv(RAW_DIR / "idd_metadata.csv", index=False)
    print(f"      Metadata saved → {RAW_DIR / 'idd_metadata.csv'}")
except Exception as e:
    print(f"      ⚠ Metadata load failed (may not have CSV files): {e}")

# ─── Option 2: Download full dataset files ────────────────────────────────────
print("\n[2/2] Downloading full IDD dataset files...")
try:
    path = kagglehub.dataset_download(
        "sovitrath/indian-driving-dataset-segmentation-part-2"
    )
    print(f"      Downloaded to : {path}")
    print(f"      Contents      : {list(Path(path).iterdir())[:10]}")

    # Copy to our data/raw directory for easy access
    dest = RAW_DIR / "idd_segmentation"
    if not dest.exists():
        shutil.copytree(path, str(dest))
        print(f"      Copied to     : {dest}")
    else:
        print(f"      Already exists: {dest}")

except Exception as e:
    print(f"      ❌ Download failed: {e}")
    print("\n💡 Manual fallback:")
    print("   1. Go to https://www.kaggle.com/datasets/sovitrath/indian-driving-dataset-segmentation-part-2")
    print("   2. Download ZIP manually")
    print("   3. Extract into data/raw/idd_segmentation/")

print("\n✅ Done. Check data/raw/ for downloaded files.")
print("Next: Convert annotations to YOLO format using a conversion script,")
print("      then copy images/labels to data/images/ and data/labels/")
