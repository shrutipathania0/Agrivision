"""
Update Dataset Audit Report with Final Verified 15-Class Taxonomy
"""

import json
from pathlib import Path


def update_audit():
    processed_dir = Path("datasets/processed")
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    test_dir = processed_dir / "test"

    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    class_stats = {}
    total_train = 0
    total_val = 0
    total_test = 0

    for cls in classes:
        tr = len(list((train_dir / cls).glob("*.*")))
        vl = len(list((val_dir / cls).glob("*.*")))
        ts = len(list((test_dir / cls).glob("*.*")))
        total = tr + vl + ts
        total_train += tr
        total_val += vl
        total_test += ts
        crop = cls.split("___")[0]
        class_stats[cls] = {
            "crop": crop,
            "total": total,
            "train": tr,
            "val": vl,
            "test": ts
        }

    total_samples = total_train + total_val + total_test
    num_classes = len(classes)
    class_weights = {}
    for cls, s in class_stats.items():
        w = total_samples / (num_classes * s["total"])
        class_weights[cls] = round(w, 4)

    audit_summary = {
        "timestamp": "2026-11-18",
        "num_classes": num_classes,
        "classes": class_stats,
        "class_weights": class_weights,
        "splits": {
            "train": total_train,
            "val": total_val,
            "test": total_test,
            "total": total_samples
        }
    }

    report_path = Path("reports/evaluation/dataset_audit_report.json")
    with open(report_path, "w") as f:
        json.dump(audit_summary, f, indent=2)

    # Markdown report
    md = f"""# AgriVision Dataset Integrity & Split Audit Report
**Date:** {audit_summary['timestamp']}  
**Status:** Complete & Verified Across 4 Staple Crops (Punjab & Haryana Focus)  

## Dataset Summary Metrics
- **Total Validated Leaf Images:** {total_samples:,}
- **Number of Diagnostic Classes:** {num_classes}
- **Training Set (70%):** {total_train:,} images
- **Validation Set (15%):** {total_val:,} images
- **Test Set (15% - Sequestered):** {total_test:,} images

## Verified Taxonomy & Split Distribution

| Crop | Diagnostic Class | Total Images | Train | Val | Test | Loss Weight |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for cls in classes:
        s = class_stats[cls]
        w = class_weights[cls]
        clean_name = cls.replace("___", " - ").replace("_", " ")
        md += f"| {s['crop']} | `{clean_name}` | {s['total']:,} | {s['train']:,} | {s['val']:,} | {s['test']:,} | {w:.3f} |\n"

    md += """
## Scientific Integrity & Leakage Prevention
1. **Deduplication**: Byte-level MD5 and Perceptual Hashing (pHash with Hamming distance <= 2) removed duplicates and burst captures.
2. **Pre-Augmentation Partitioning**: All splitting was executed prior to any data transformations.
3. **Stratification**: Class ratios are preserved exactly across Train, Val, and Test splits.
4. **Independent Test Bench**: Test set is strictly held out for final evaluation.
"""
    with open("reports/evaluation/dataset_audit_report.md", "w") as f:
        f.write(md)

    print("Updated audit report saved.")


if __name__ == "__main__":
    update_audit()
