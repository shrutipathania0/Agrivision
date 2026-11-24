"""
AgriVision Automated Dataset Downloader & Taxonomy Structurer
Downloads verified subsets for:
  - Potato (Early Blight, Late Blight, Healthy)
  - Tomato (Early Blight, Late Blight, Septoria, Leaf Mold, Bacterial Spot, Mosaic Virus, Healthy)
  - Rice (Bacterial Blight, Brown Spot, Leaf Smut)
  - Wheat (Yellow Rust, Healthy)
  - PlantDoc (External Real-World Test Bench)
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


RAW_DIR = Path("datasets/raw").resolve()
PROCESSED_DIR = Path("datasets/processed").resolve()
EXTERNAL_DIR = Path("datasets/external_test").resolve()


REPOS = {
    "potato": {
        "url": "https://github.com/Vishal2546/Potato-Disease-Classification.git",
        "path": RAW_DIR / "potato_repo",
        "source_map": {
            "PlantVillage/Potato___Early_blight": "Potato___Early_Blight",
            "PlantVillage/Potato___Late_blight": "Potato___Late_Blight",
            "PlantVillage/Potato___healthy": "Potato___Healthy",
        }
    },
    "tomato": {
        "url": "https://github.com/bidya108/Tomato-blight-detection.git",
        "path": RAW_DIR / "tomato_repo",
        "source_map": {
            "plantvillage/Tomato___Early_blight": "Tomato___Early_Blight",
            "plantvillage/Tomato___Late_blight": "Tomato___Late_Blight",
            "plantvillage/Tomato___Septoria_leaf_spot": "Tomato___Septoria_Leaf_Spot",
            "plantvillage/Tomato___Leaf_Mold": "Tomato___Leaf_Mold",
            "plantvillage/Tomato___Bacterial_spot": "Tomato___Bacterial_Spot",
            "plantvillage/Tomato___Tomato_mosaic_virus": "Tomato___Mosaic_Virus",
            "plantvillage/Tomato___healthy": "Tomato___Healthy",
        }
    },
    "rice": {
        "url": "https://github.com/MHassaanButt/Rice-Disease-Classfication.git",
        "path": RAW_DIR / "rice_repo",
        "source_map": {
            "Dataset/Bacterial leaf blight": "Rice___Bacterial_Leaf_Blight",
            "Dataset/Brown spot": "Rice___Brown_Spot",
            "Dataset/Leaf smut": "Rice___Leaf_Smut",
        }
    },
    "wheat": {
        "url": "https://github.com/Shant-Thakur/YR-22-23.git",
        "path": RAW_DIR / "wheat_repo",
        "source_map": {
            "dataset/train/rust": "Wheat___Yellow_Rust",
            "dataset/test/rust": "Wheat___Yellow_Rust",
            "dataset/train/healthy": "Wheat___Healthy",
            "dataset/test/healthy": "Wheat___Healthy",
        }
    }
}


def clone_repo(name, url, target_path):
    if target_path.exists() and any(target_path.iterdir()):
        print(f"[{name}] Already exists at {target_path}. Skipping clone.")
        return True
    print(f"[{name}] Shallow cloning from {url} to {target_path}...")
    cmd = ["git", "clone", "--depth", "1", url, str(target_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[{name}] Clone failed: {result.stderr}")
        return False
    print(f"[{name}] Cloned successfully.")
    return True


def clone_plantdoc():
    target_path = RAW_DIR / "plantdoc_repo"
    if target_path.exists() and any(target_path.iterdir()):
        print(f"[PlantDoc] Already exists at {target_path}. Skipping clone.")
        return True
    print(f"[PlantDoc] Cloning test subset from pratikkayal/PlantDoc-Dataset...")
    # Clone with filter to avoid huge download
    cmd = ["git", "clone", "--depth", "1", "https://github.com/pratikkayal/PlantDoc-Dataset.git", str(target_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[PlantDoc] Clone failed: {result.stderr}")
        return False
    print(f"[PlantDoc] Cloned successfully.")
    return True


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AGRIVISION DATASET ACQUISITION PIPELINE")
    print("=" * 60)

    for name, config in REPOS.items():
        success = clone_repo(name, config["url"], config["path"])
        if not success:
            print(f"[WARN] Failed to clone {name}. Training will proceed with available sets.")

    print("\nFetching PlantDoc external test benchmark...")
    clone_plantdoc()

    print("\n" + "=" * 60)
    print("Dataset acquisition phase complete. Ready for integrity audit.")
    print("=" * 60)


if __name__ == "__main__":
    main()
