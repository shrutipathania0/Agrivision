"""
AgriVision Direct Wheat Rust Ingestion & Split
Downloads genuine Wheat Leaf Rust images directly from repository raw assets,
splits them 70/15/15, and updates class mapping.
"""

import os
import shutil
import urllib.request
import urllib.parse
import json
from pathlib import Path
import numpy as np


RAW_TARGET = Path("datasets/raw/wheat_leaf_rust").resolve()
PROCESSED_DIR = Path("datasets/processed").resolve()


def fetch_wheat_rust():
    RAW_TARGET.mkdir(parents=True, exist_ok=True)
    api_url = "https://api.github.com/repos/aadium/wheat-disease-detection/contents/cropDiseaseDataset/Wheat_Leaf_Rust"
    req = urllib.request.Request(api_url, headers={"User-Agent": "AgriVision-Auditor"})

    print("Fetching Wheat_Leaf_Rust file list from GitHub API...")
    with urllib.request.urlopen(req) as resp:
        items = json.loads(resp.read().decode())

    image_files = [x for x in items if x["type"] == "file" and x["name"].lower().endswith((".jpg", ".jpeg", ".png", ".jfif"))]
    print(f"Found {len(image_files)} genuine Wheat Leaf Rust images. Downloading...")

    downloaded = []
    for idx, item in enumerate(image_files):
        filename = item["name"]
        raw_url = item["download_url"]
        dest = RAW_TARGET / f"wheat_rust_{idx:04d}.jpg"
        try:
            req_dl = urllib.request.Request(raw_url, headers={"User-Agent": "AgriVision"})
            with urllib.request.urlopen(req_dl) as r, open(dest, "wb") as f:
                f.write(r.read())
            downloaded.append(dest)
        except Exception as e:
            print(f"  Error downloading {filename}: {e}")

    print(f"Downloaded {len(downloaded)} images successfully.")

    # Remove the 2-image Yellow Rust folder if present to avoid underfitted classes
    for split in ["train", "val", "test"]:
        yr_dir = PROCESSED_DIR / split / "Wheat___Yellow_Rust"
        if yr_dir.exists():
            shutil.rmtree(yr_dir)

    # Stratified 70/15/15 split for Wheat___Leaf_Rust
    n = len(downloaded)
    indices = np.random.RandomState(42).permutation(n)
    tr_end = int(0.70 * n)
    vl_end = int(0.85 * n)

    splits = {
        "train": [downloaded[i] for i in indices[:tr_end]],
        "val": [downloaded[i] for i in indices[tr_end:vl_end]],
        "test": [downloaded[i] for i in indices[vl_end:]]
    }

    cls_name = "Wheat___Leaf_Rust"
    for split_name, paths in splits.items():
        dest_dir = PROCESSED_DIR / split_name / cls_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(paths):
            shutil.copy2(p, dest_dir / f"{cls_name}_{i:04d}.jpg")

    print(f"Wheat Leaf Rust split complete: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    # Update class mapping for all existing classes
    train_dir = PROCESSED_DIR / "train"
    all_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    mapping = {
        "classes": all_classes,
        "num_classes": len(all_classes),
        "class_to_idx": {cls: i for i, cls in enumerate(all_classes)},
        "idx_to_class": {i: cls for i, cls in enumerate(all_classes)}
    }
    with open("models/computer_vision/class_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"Updated class_mapping.json with {len(all_classes)} verified classes:")
    for c in all_classes:
        print(f"  - {c}")


if __name__ == "__main__":
    fetch_wheat_rust()
