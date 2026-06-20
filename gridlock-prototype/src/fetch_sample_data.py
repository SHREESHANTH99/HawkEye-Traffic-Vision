"""
fetch_sample_data.py
Downloads sample Indian traffic images from public sources into data/raw/samples/
so the Streamlit app has demo images to work with immediately.
"""
import urllib.request
import os
from pathlib import Path

SAMPLE_DIR = Path("data/raw/samples")
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

# Public domain / CC0 traffic images (direct URLs to raw image files)
IMAGES = [
    # Indian traffic scenes - public Wikimedia Commons images
    ("traffic_india_1.jpg",  "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/India_traffic_2009.jpg/800px-India_traffic_2009.jpg"),
    ("traffic_india_2.jpg",  "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Bangalore_traffic.jpg/800px-Bangalore_traffic.jpg"),
    ("traffic_india_3.jpg",  "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Mumbai_traffic_1.jpg/800px-Mumbai_traffic_1.jpg"),
    ("motorcycle_india.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/Two_wheelers_in_India.jpg/800px-Two_wheelers_in_India.jpg"),
]

print("Downloading sample traffic images...")
for fname, url in IMAGES:
    dest = SAMPLE_DIR / fname
    if dest.exists():
        print(f"  [SKIP] {fname} already exists")
        continue
    try:
        urllib.request.urlretrieve(url, str(dest))
        size = dest.stat().st_size
        print(f"  [OK]   {fname}  ({size//1024} KB)")
    except Exception as e:
        print(f"  [ERR]  {fname}: {e}")

print(f"\nSample images saved to: {SAMPLE_DIR.resolve()}")
print("You can now upload these in the Streamlit app for demo.")
