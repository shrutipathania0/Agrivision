# AgriVision System Architecture & Engineering Specifications

## 1. Architectural Overview
AgriVision is an end-to-end agricultural intelligence application combining computer vision, natural language processing, and semantic information retrieval to assist farmers and agricultural students in Punjab and Haryana, India.

```
+-------------------------------------------------------------------------+
|                  Client Interface (React + Vite + Tailwind)             |
|   - Image Dropzone & Camera Input      - Grad-CAM Heatmap Visualizer   |
|   - Dynamic Advisory Cards             - Context-Seeded Chatbot         |
|   - PDF Report Generator (Blob)        - Supported Taxonomy Guide       |
+------------------------------------+------------------------------------+
                                     |
                                     | HTTP REST (JSON / Multipart)
                                     v
+-------------------------------------------------------------------------+
|                       FastAPI Gateway (Uvicorn)                         |
|   POST /api/detect  |  POST /api/chat  |  POST /api/report  |  /health  |
+------------------------------------+------------------------------------+
                                     |
       +-----------------------------+-----------------------------+
       |                                                           |
       v                                                           v
+-----------------------------+             +-----------------------------+
|    Computer Vision Engine   |             |     NLP Advisory Engine     |
| - Preprocessing (224x224)   |             | - DistilBERT (6 Intents)    |
| - EfficientNet-B0 (PyTorch) |             | - all-MiniLM-L6-v2 Embedder |
| - Shannon Entropy Filter    |             | - FAISS IndexFlatIP Cosine  |
| - Grad-CAM Visual Attention |             | - ICAR/PAU/HAU Knowledge DB |
+-----------------------------+             +-----------------------------+
```

## 2. Computer Vision Subsystem
- **Base Architecture**: EfficientNet-B0 (ImageNet Pretrained).
- **Taxonomy**: 15 classes covering 4 staple crops (Potato, Tomato, Rice, Wheat).
- **Training Strategy**: Two-Stage Transfer Learning:
  - *Stage 1*: Backbone frozen, linear projection head trained (5 epochs).
  - *Stage 2*: Top convolutional stages (`features[6]`, `features[7]`) unfrozen with differential learning rates (`1e-4` backbone, `5e-4` head) and `ReduceLROnPlateau`.
- **Explainability**: Grad-CAM registered at layer `features[-1]`, projecting normalized activation weights onto original leaf photographs.
- **Uncertainty & Out-of-Distribution Handling**: Softmax confidence threshold ($0.65$) combined with Shannon entropy threshold ($1.80$) flags ambiguous or non-leaf uploads.

## 3. Natural Language Processing Subsystem
- **Intent Classifier**: Pretrained `distilbert-base-uncased` fine-tuned across 6 agricultural advisory intents:
  1. `disease_symptoms`
  2. `disease_causes`
  3. `disease_prevention`
  4. `disease_management`
  5. `fertilizer_nutrient_info`
  6. `general_farming_inquiry`
- **Semantic Vector Retrieval**: `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional dense embeddings stored in a normalized `faiss.IndexFlatIP` index.
- **Context Injection**: Computer vision predictions (crop, disease, confidence) are dynamically bound to user sessions, prioritizing contextual advisories without persistent storage of user data.

## 4. Grounded Knowledge Sources
All agronomic advisories, cultural practices, and safety instructions are derived from verified institutional authorities:
- Punjab Agricultural University (PAU), Ludhiana
- Chaudhary Charan Singh Haryana Agricultural University (CCS HAU), Hisar
- ICAR - Indian Institute of Wheat and Barley Research (IIWBR), Karnal
- ICAR - Central Potato Research Institute (CPRI), Shimla
- ICAR - National Rice Research Institute (NRRI), Cuttack
- Krishi Vigyan Kendra (KVK) Network
