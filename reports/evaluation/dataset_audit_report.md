# AgriVision Dataset Integrity & Split Audit Report
**Date:** 2026-11-18  
**Status:** Complete & Verified Across 4 Staple Crops (Punjab & Haryana Focus)  

## Dataset Summary Metrics
- **Total Validated Leaf Images:** 11,364
- **Number of Diagnostic Classes:** 15
- **Training Set (70%):** 7,950 images
- **Validation Set (15%):** 1,704 images
- **Test Set (15% - Sequestered):** 1,710 images

## Verified Taxonomy & Split Distribution

| Crop | Diagnostic Class | Total Images | Train | Val | Test | Loss Weight |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Potato | `Potato - Early Blight` | 999 | 699 | 150 | 150 | 0.758 |
| Potato | `Potato - Healthy` | 152 | 106 | 23 | 23 | 4.984 |
| Potato | `Potato - Late Blight` | 1,000 | 700 | 150 | 150 | 0.758 |
| Rice | `Rice - Bacterial Leaf Blight` | 40 | 28 | 6 | 6 | 18.940 |
| Rice | `Rice - Brown Spot` | 40 | 28 | 6 | 6 | 18.940 |
| Rice | `Rice - Leaf Smut` | 40 | 28 | 6 | 6 | 18.940 |
| Tomato | `Tomato - Bacterial Spot` | 1,702 | 1,191 | 255 | 256 | 0.445 |
| Tomato | `Tomato - Early Blight` | 800 | 560 | 120 | 120 | 0.947 |
| Tomato | `Tomato - Healthy` | 1,268 | 887 | 190 | 191 | 0.598 |
| Tomato | `Tomato - Late Blight` | 1,522 | 1,065 | 228 | 229 | 0.498 |
| Tomato | `Tomato - Leaf Mold` | 761 | 532 | 114 | 115 | 0.996 |
| Tomato | `Tomato - Mosaic Virus` | 299 | 209 | 45 | 45 | 2.534 |
| Tomato | `Tomato - Septoria Leaf Spot` | 1,417 | 991 | 213 | 213 | 0.535 |
| Wheat | `Wheat - Healthy` | 1,189 | 832 | 178 | 179 | 0.637 |
| Wheat | `Wheat - Leaf Rust` | 135 | 94 | 20 | 21 | 5.612 |

## Scientific Integrity & Leakage Prevention
1. **Deduplication**: Byte-level MD5 and Perceptual Hashing (pHash with Hamming distance <= 2) removed duplicates and burst captures.
2. **Pre-Augmentation Partitioning**: All splitting was executed prior to any data transformations.
3. **Stratification**: Class ratios are preserved exactly across Train, Val, and Test splits.
4. **Independent Test Bench**: Test set is strictly held out for final evaluation.
