# AgriVision Model Evaluation & Benchmark Report
**Date:** 2026-11-18  
**Evaluation Set:** Sequestered Test Split (1,942 images)  
**Architecture:** ImageNet Pretrained EfficientNet-B0 (Two-Stage Transfer Learning)  
**Inference Device:** CUDA  

## Overall Performance Metrics

| Metric | Measured Value | Acceptance Target | Evaluation Result |
| :--- | :--- | :--- | :--- |
| **Top-1 Test Accuracy** | **94.64%** | $\ge 90.0\%$ | PASSED |
| **Macro-F1 Score** | **0.9124** | $\ge 0.8500$ | PASSED |
| **Weighted-F1 Score** | **0.9473** | $\ge 0.8500$ | PASSED |
| **Macro Precision** | **0.9066** | $\ge 0.8500$ | PASSED |
| **Macro Recall** | **0.9213** | $\ge 0.8500$ | PASSED |
| **Expected Calibration Error (ECE)** | **0.0180** | $\le 0.1000$ | WELL CALIBRATED |
| **Mean Inference Latency** | **18.56 ms** | $\le 5,000$ ms | HIGH-PERFORMANCE (<20ms) |
| **P95 Latency** | **18.89 ms** | $\le 5,000$ ms | HIGH-PERFORMANCE |

## Per-Crop Accuracy Summary

| Crop | Test Samples | Accuracy | Status |
| :--- | :--- | :--- | :--- |
| **Potato** | 359 | **93.6%** | Robust High Accuracy |
| **Tomato** | 1282 | **94.7%** | Robust High Accuracy |
| **Rice** | 48 | **89.6%** | Validated |
| **Wheat** | 200 | **98.5%** | Validated |

## Per-Class Precision, Recall & F1 Breakdown

| Crop | Diagnostic Class | Test Images | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Non_Target | `Non Target - Background or Other Leaf` | 53.0 | 0.696 | 0.906 | 0.787 |
| Potato | `Potato - Early Blight` | 169.0 | 0.945 | 0.923 | 0.934 |
| Potato | `Potato - Healthy` | 23.0 | 0.958 | 1.000 | 0.979 |
| Potato | `Potato - Late Blight` | 167.0 | 0.975 | 0.940 | 0.957 |
| Rice | `Rice - Bacterial Leaf Blight` | 21.0 | 0.875 | 1.000 | 0.933 |
| Rice | `Rice - Brown Spot` | 21.0 | 0.900 | 0.857 | 0.878 |
| Rice | `Rice - Leaf Smut` | 6.0 | 0.667 | 0.667 | 0.667 |
| Tomato | `Tomato - Bacterial Spot` | 274.0 | 0.974 | 0.953 | 0.963 |
| Tomato | `Tomato - Early Blight` | 135.0 | 0.920 | 0.933 | 0.926 |
| Tomato | `Tomato - Healthy` | 202.0 | 0.985 | 0.960 | 0.972 |
| Tomato | `Tomato - Late Blight` | 248.0 | 0.952 | 0.964 | 0.958 |
| Tomato | `Tomato - Leaf Mold` | 130.0 | 0.924 | 0.938 | 0.931 |
| Tomato | `Tomato - Mosaic Virus` | 55.0 | 0.820 | 0.909 | 0.862 |
| Tomato | `Tomato - Septoria Leaf Spot` | 238.0 | 0.978 | 0.933 | 0.955 |
| Wheat | `Wheat - Healthy` | 179.0 | 0.989 | 1.000 | 0.994 |
| Wheat | `Wheat - Leaf Rust` | 21.0 | 0.947 | 0.857 | 0.900 |

## Visual Saliency & Confusion Matrix
- **Confusion Matrix Graphic**: [cv_confusion_matrix.png](cv_confusion_matrix.png)
- **Grad-CAM Integration**: Verified with target feature layer `features[-1]`.
