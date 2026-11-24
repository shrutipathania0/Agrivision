# AgriVision User & Operator Guide
**Smart Crop Disease Detection, Agricultural Advisory & AI Assistant**

---

## 1. Quick Launch
To start both the FastAPI backend and React frontend simultaneously:

```powershell
# Open terminal in project root
cd d:\Code\Code\AgriBot

# Launch both servers
.\.venv\Scripts\python.exe run_agrivision.py
```

Once running:
- **Interactive Web Interface**: Open [http://localhost:5173](http://localhost:5173) in Chrome or Edge.
- **Interactive REST API Documentation**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
- **Backend Health Check**: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health).

---

## 2. Using the Web Dashboard

### A. Uploading Leaf Imagery
1. Drag and drop any leaf photograph onto the dashed dropzone, or click **"Select File"**.
2. On smartphones or laptops equipped with webcams, click **"Use Camera"** to capture live leaf foliage.
3. Review the live image preview and resolution badge (recommended $\ge 224 \times 224$ pixels).
4. Click **"Run AI Disease Diagnosis"**.

### B. Interpreting Diagnostics & Grad-CAM
- **Status Badge**: Indicates whether foliage is classified as **Healthy** (Emerald badge) or **Diseased** (Rose badge).
- **AI Confidence**: Softmax probability percentage. Scores below 65% trigger an uncertainty warning asking the user to review the image.
- **Grad-CAM Heatmap Visualizer**:
  - Click **"Heatmap ON"** to toggle visual attention overlay.
  - Adjust the **Heatmap Blend Slider** (0% to 100%) to inspect the exact leaf lesions that triggered the neural network's decision.
- **Alternative Predictions**: Displays top-4 class probabilities for differential diagnoses.

### C. Agronomic Advisory & Recommendations
- **Diagnostic Symptoms**: Detailed physical manifestations of the disease.
- **Preventive Agronomic Measures**: Certified seeds, seed treatment, crop rotation, spacing.
- **Cultural Sanitation & Field Control**: Clean field margins, rogueing, irrigation adjustments.
- **Mandatory Safety Notice**: Safe pesticide handling, personal protective equipment (PPE), and warning to consult local Krishi Vigyan Kendra (KVK) officers.

### D. Interactive Farmer Chatbot
- Type questions naturally into the chat input bar (e.g. *"Can this blight spread to neighboring potato crops?"*).
- The chatbot automatically inherits the current scan's crop and disease context to deliver personalized, grounded answers.
- Click any of the quick-prompt suggestion chips for rapid advisory retrieval.
- Institutional sources (PAU, CCS HAU, ICAR) are cited directly on each response card.

### E. Downloadable PDF Report
- Click the **"PDF Report"** button in the diagnosis banner.
- AgriVision generates and downloads a branded, printable diagnostic summary including:
  - Analysis timestamp and diagnostic status
  - High-resolution Grad-CAM saliency map
  - Scientific pathogen classification
  - Integrated management instructions
  - Official ICAR agricultural safety disclaimers

---

## 3. Supported Crops & Pathologies

| Crop | Supported Conditions |
| :--- | :--- |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Tomato** | Early Blight, Late Blight, Septoria Leaf Spot, Leaf Mold, Bacterial Spot, Mosaic Virus, Healthy |
| **Rice** | Bacterial Leaf Blight, Blast, Brown Spot, Leaf Smut |
| **Wheat** | Leaf Rust (Brown Rust), Yellow Rust (Stripe Rust), Healthy |

---

## 4. Verification & Testing
To execute the automated regression test suite:

```powershell
.\.venv\Scripts\pytest.exe tests/test_pipeline.py -v
```
