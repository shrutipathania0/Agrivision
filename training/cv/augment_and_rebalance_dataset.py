"""
AgriVision Real-World Dataset Augmentation & Rebalancing Engine
1. Merges PlantDoc outdoor field images into Potato and Tomato classes.
2. Merges AveyBD outdoor field images into Rice classes.
3. Builds 16th class: Non_Target___Background_or_Other_Leaf.
4. Splits all additions stratified 70/15/15 into train/val/test.
5. Recomputes smoothed, balanced class weights (capped at [0.5, 2.5]).
6. Exports updated class mapping and audit report.
"""

import os
import shutil
import json
import random
from pathlib import Path
from PIL import Image
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROCESSED_DIR = Path("datasets/processed").resolve()
PLANTDOC_DIR = Path("datasets/raw/plantdoc_extracted").resolve()
RICE_EXT_DIR = Path("datasets/raw/rice_extracted").resolve()
REPORTS_DIR = Path("reports/evaluation").resolve()
MODELS_DIR = Path("models/computer_vision").resolve()

# 1. Map PlantDoc folders to our taxonomy
PLANTDOC_MAP = {
    "Potato___Early_Blight": [
        "train_Potato leaf early blight", "test_Potato leaf early blight"
    ],
    "Potato___Late_Blight": [
        "train_Potato leaf late blight", "test_Potato leaf late blight"
    ],
    "Tomato___Early_Blight": [
        "train_Tomato Early blight leaf", "test_Tomato Early blight leaf"
    ],
    "Tomato___Late_Blight": [
        "train_Tomato leaf late blight", "test_Tomato leaf late blight"
    ],
    "Tomato___Bacterial_Spot": [
        "train_Tomato leaf bacterial spot", "test_Tomato leaf bacterial spot"
    ],
    "Tomato___Leaf_Mold": [
        "train_Tomato mold leaf", "test_Tomato mold leaf"
    ],
    "Tomato___Mosaic_Virus": [
        "train_Tomato leaf mosaic virus", "test_Tomato leaf mosaic virus"
    ],
    "Tomato___Septoria_Leaf_Spot": [
        "train_Tomato Septoria leaf spot", "test_Tomato Septoria leaf spot"
    ],
    "Tomato___Healthy": [
        "train_Tomato leaf", "test_Tomato leaf"
    ]
}

# 2. Map Rice folders
RICE_MAP = {
    "Rice___Bacterial_Leaf_Blight": [
        RICE_EXT_DIR / "rice" / "train" / "bacterial_leaf_blight",
        RICE_EXT_DIR / "rice" / "validation" / "bacterial_leaf_blight"
    ],
    "Rice___Brown_Spot": [
        RICE_EXT_DIR / "rice" / "train" / "brownspot",
        RICE_EXT_DIR / "rice" / "validation" / "brownspot"
    ]
}

# 3. Non-Target folders for the 16th class
NON_TARGET_PLANTDOC_FOLDERS = [
    "train_grape leaf", "train_grape leaf black rot", "test_grape leaf",
    "train_Peach leaf", "test_Peach leaf",
    "train_Apple leaf", "train_Apple rust leaf",
    "train_Strawberry leaf",
    "train_Bell_pepper leaf"
]

def split_and_copy(file_list, class_name, prefix="ext"):
    valid_files = []
    for f in file_list:
        try:
            with Image.open(f) as im:
                im.verify()
            valid_files.append(f)
        except Exception:
            continue

    n = len(valid_files)
    if n == 0:
        return 0, 0, 0

    indices = np.random.permutation(n)
    tr_end = int(0.70 * n)
    vl_end = int(0.85 * n)

    train_files = [valid_files[i] for i in indices[:tr_end]]
    val_files = [valid_files[i] for i in indices[tr_end:vl_end]]
    test_files = [valid_files[i] for i in indices[vl_end:]]

    for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        dest_dir = PROCESSED_DIR / split_name / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(files):
            dest_name = f"{prefix}_{src.stem[:12]}_{i:04d}{src.suffix}"
            shutil.copy2(src, dest_dir / dest_name)

    return len(train_files), len(val_files), len(test_files)

def main():
    print("=" * 70)
    print("INGESTING REAL-WORLD FIELD IMAGES & NON-TARGET BACKGROUND CLASS")
    print("=" * 70)

    # Clean any prior extension files to allow re-runs
    for split in ["train", "val", "test"]:
        s_dir = PROCESSED_DIR / split
        if s_dir.exists():
            for f in s_dir.rglob("pdoc_*.*"):
                f.unlink(missing_ok=True)
            for f in s_dir.rglob("riceext_*.*"):
                f.unlink(missing_ok=True)
            for f in s_dir.rglob("nontarget_*.*"):
                f.unlink(missing_ok=True)
            # Remove old Non_Target folder if present
            nt_dir = s_dir / "Non_Target___Background_or_Other_Leaf"
            if nt_dir.exists():
                shutil.rmtree(nt_dir)

    # A. Ingest PlantDoc field images
    print("\n1. Ingesting PlantDoc outdoor field images for Potato & Tomato...")
    for target_class, folders in PLANTDOC_MAP.items():
        all_files = []
        for fld in folders:
            src_path = PLANTDOC_DIR / fld
            if src_path.exists():
                all_files.extend(list(src_path.glob("*.jpg")))
                all_files.extend(list(src_path.glob("*.png")))
        tr, vl, ts = split_and_copy(all_files, target_class, prefix="pdoc")
        print(f"  + {target_class:30s} -> Added Train: {tr}, Val: {vl}, Test: {ts}")

    # B. Ingest Rice outdoor field images
    print("\n2. Ingesting Rice outdoor field images...")
    for target_class, paths in RICE_MAP.items():
        all_files = []
        for p in paths:
            if p.exists():
                all_files.extend(list(p.glob("*.jpg")))
                all_files.extend(list(p.glob("*.png")))
        tr, vl, ts = split_and_copy(all_files, target_class, prefix="riceext")
        print(f"  + {target_class:30s} -> Added Train: {tr}, Val: {vl}, Test: {ts}")

    # C. Build Non_Target 16th class
    print("\n3. Building 16th Class: Non_Target___Background_or_Other_Leaf...")
    non_target_files = []
    for fld in NON_TARGET_PLANTDOC_FOLDERS:
        src_path = PLANTDOC_DIR / fld
        if src_path.exists():
            non_target_files.extend(list(src_path.glob("*.jpg")))
            non_target_files.extend(list(src_path.glob("*.png")))

    # Cap to ~350 images to keep balance
    random.shuffle(non_target_files)
    selected_nt = non_target_files[:350]
    tr, vl, ts = split_and_copy(selected_nt, "Non_Target___Background_or_Other_Leaf", prefix="nontarget")
    print(f"  + Non_Target class created -> Train: {tr}, Val: {vl}, Test: {ts}")

    # D. Recalculate dataset distribution and balanced loss weights
    print("\n4. Auditing final augmented dataset distribution...")
    train_dir = PROCESSED_DIR / "train"
    all_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    counts = {}
    total_train = 0

    for cls in all_classes:
        n_tr = len(list((PROCESSED_DIR / "train" / cls).glob("*.*")))
        n_vl = len(list((PROCESSED_DIR / "val" / cls).glob("*.*")))
        n_ts = len(list((PROCESSED_DIR / "test" / cls).glob("*.*")))
        counts[cls] = {"train": n_tr, "val": n_vl, "test": n_ts, "total": n_tr + n_vl + n_ts}
        total_train += n_tr

    num_classes = len(all_classes)
    avg_per_class = total_train / num_classes

    # Smoothed loss weights: w_c = sqrt(avg / n_c), clipped to [0.5, 2.5]
    class_weights = {}
    for cls in all_classes:
        n_c = counts[cls]["train"]
        raw_w = np.sqrt(avg_per_class / max(n_c, 1))
        clipped_w = float(np.clip(raw_w, 0.5, 2.5))
        class_weights[cls] = round(clipped_w, 4)

    print("\nFinal Class Counts & Smoothed Weights:")
    for cls in all_classes:
        print(f"  {cls:40s} | Total: {counts[cls]['total']:4d} (Train: {counts[cls]['train']:4d}) | Weight: {class_weights[cls]:.4f}")

    # E. Export updated class mapping
    mapping = {
        "classes": all_classes,
        "num_classes": num_classes,
        "class_to_idx": {cls: i for i, cls in enumerate(all_classes)},
        "idx_to_class": {i: cls for i, cls in enumerate(all_classes)}
    }
    with open(MODELS_DIR / "class_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)

    audit_report = {
        "timestamp": "2026-11-18",
        "num_classes": num_classes,
        "classes": counts,
        "class_weights": class_weights,
        "splits": {
            "train": sum(c["train"] for c in counts.values()),
            "val": sum(c["val"] for c in counts.values()),
            "test": sum(c["test"] for c in counts.values()),
            "total": sum(c["total"] for c in counts.values())
        }
    }
    with open(REPORTS_DIR / "dataset_audit_report.json", "w") as f:
        json.dump(audit_report, f, indent=2)

    print(f"\n[OK] Updated class_mapping.json and dataset_audit_report.json with {num_classes} classes.")
    print(f"Total dataset size: {audit_report['splits']['total']} images.")

if __name__ == "__main__":
    main()
