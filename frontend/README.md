# 🌱 AgriVision – Smart Crop Disease Detection & Agricultural Advisory System

## 1. About the Project

AgriVision is an AI-powered web application developed to help farmers identify crop diseases and access useful agricultural information.

The application combines Artificial Intelligence, Computer Vision and Natural Language Processing (NLP) to provide two major services:

- **Crop Disease Detection:** Identifies crop diseases by analysing uploaded leaf images.
- **Agricultural Assistance:** Provides agricultural information and answers farming-related questions through an AI-powered chatbot.

The initial version focuses on four major crops: Wheat, Rice, Potato and Tomato, with a geographical focus on Punjab and Haryana.

The main aim of AgriVision is to make agricultural information more accessible and help farmers identify possible crop diseases at an early stage.

## 2. Problem Statement

Crop diseases can reduce agricultural production and cause financial losses for farmers.

Farmers may face difficulties in identifying diseases, understanding their symptoms and finding reliable information about disease management.

AgriVision aims to address these problems through an easy-to-use platform that combines image-based disease detection with agricultural information and assistance.

## 3. Project Objectives

- Develop an AI-based crop disease detection system.
- Identify healthy and diseased crop leaves.
- Support multiple crop categories.
- Display prediction confidence.
- Provide visual explanations of predictions.
- Provide disease information and management suggestions.
- Develop an agricultural chatbot.
- Retrieve relevant information from a structured knowledge base.
- Generate downloadable PDF reports.

## 4. Key Features

### 1. Crop Disease Detection
Users can upload a crop leaf image, and the application identifies the supported crop and its possible disease using a trained deep learning model.

### 2. Model Evaluation
The application provides a separate evaluation dashboard displaying the performance of the trained model through evaluation metrics and visualizations.

### 3. Grad-CAM Visualization
Grad-CAM helps explain model predictions by highlighting image regions that influenced the model's decision. It makes the prediction process easier to understand.

### 4. Agricultural Chatbot
The chatbot allows users to ask questions about crop diseases, symptoms, causes, prevention and general disease management.

### 5. Agricultural Knowledge Base
The application maintains a collection of agricultural information from reliable sources. It helps the chatbot retrieve relevant information for users.

### 6. PDF Report Generation
Users can generate downloadable reports containing disease predictions, confidence scores, image information, visual explanations and agricultural recommendations.

### 7. User-Friendly Interface
The application provides a simple and responsive interface so users can access its features easily.

## 5. Supported Crops

AgriVision initially supports four major crops:

| Crop | Supported Disease Categories |
|---|---|
| Wheat | Healthy, Leaf Rust |
| Rice | Healthy, Bacterial Leaf Blight, Brown Spot, Leaf Smut |
| Potato | Healthy, Early Blight, Late Blight |
| Tomato | Healthy, Early Blight, Late Blight, Bacterial Spot, Leaf Mold, Mosaic Virus, Septoria Leaf Spot |

The final supported categories depend on the verified dataset and trained model.

## 6. Technology Stack

### A. Frontend Development

| Technology | Purpose |
|---|---|
| React.js | Used to build the interactive user interface. |
| Vite | Provides a fast development environment and builds the frontend application. |
| Tailwind CSS | Used to style the application and create a responsive design. |

### B. Backend Development

| Technology | Purpose |
|---|---|
| Python | Main programming language used for AI models and backend development. |
| FastAPI | Used to create APIs and connect the frontend with the backend. |
| Uvicorn | Runs the FastAPI application as a web server. |
| Pydantic | Validates data received and returned by the APIs. |
| Pydantic-settings | Manages application configuration and settings. |
| Python-multipart | Handles image files uploaded by users. |

### C. Computer Vision & Image Processing

| Technology | Purpose |
|---|---|
| PyTorch | Deep learning framework used to train and run AI models. |
| EfficientNet-B0 | Pretrained image classification architecture used for crop disease detection. |
| OpenCV | Used for image processing and manipulation. |
| Pillow | Used to open, read, resize and convert images. |
| NumPy | Used for numerical calculations and image arrays. |
| ImageHash | Helps identify duplicate or visually similar images. |

### D. Natural Language Processing (NLP)

| Technology | Purpose |
|---|---|
| Hugging Face Transformers | Provides pretrained NLP models and tools for fine-tuning them. |
| DistilBERT | Used to classify agricultural questions into different intents. |
| Sentence Transformers | Converts text into numerical embeddings for semantic similarity. |
| FAISS | Searches for relevant information using vector similarity. |
| Accelerate | Helps manage efficient model execution and training on available hardware. |

### E. Data Analysis & Visualization

| Technology | Purpose |
|---|---|
| Pandas | Used to organize and analyse dataset information. |
| Scikit-learn | Used to calculate machine learning evaluation metrics. |
| Matplotlib | Used to create graphs and visualize model results. |
| Seaborn | Used to create statistical visualizations such as confusion matrix heatmaps. |

### F. Additional Python Libraries

| Technology | Purpose |
|---|---|
| ReportLab | Generates downloadable PDF reports. |
| tqdm | Displays progress bars during long-running tasks. |
| PyYAML | Reads configuration settings from YAML files. |
| Requests | Allows communication with external APIs and web services. |
| Pytest | Used for automated testing of Python code. |
| HTTPX | Sends HTTP requests, particularly for testing backend APIs. |

## 7. Dataset Sources

AgriVision uses publicly available agricultural image datasets for model development and evaluation.

| Dataset | Purpose |
|---|---|
| PlantVillage | Provides healthy and diseased Potato and Tomato leaf images. |
| Rice Leaf Disease Dataset | Provides images of different Rice diseases. |
| Wheat Fungi Diseases Dataset | Provides healthy and diseased Wheat images. |
| PlantDoc | Provides more natural plant images for external model evaluation. |

## 8. Model Evaluation

Model evaluation helps us understand how accurately our trained AI model performs on unseen test data.

AgriVision uses the following evaluation metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures the overall percentage of correct predictions. |
| Precision | Measures how many predicted disease cases are actually correct. |
| Recall | Measures how many actual disease cases are correctly identified. |
| Macro-F1 Score | Measures the average F1-score across all classes, giving equal importance to each class. |
| Confusion Matrix | Displays correct and incorrect predictions for individual disease categories. |

These metrics help us understand the strengths and weaknesses of the model and identify areas that need improvement.

## 9. Agricultural Knowledge Sources

AgriVision aims to provide reliable agricultural information using institutional sources such as:

- Indian Council of Agricultural Research (ICAR).
- Punjab Agricultural University (PAU).
- CCS Haryana Agricultural University (CCS HAU).
- Krishi Vigyan Kendra (KVK).

The knowledge base contains information about crop diseases, symptoms, causes, prevention and general management practices.

## 10. Future Scope

The project can be improved further by introducing:

- Regional language support.
- Voice-based agricultural assistance.
- Support for additional crops.
- Weather information integration.
- Improved disease severity estimation.
- Mobile application development.
- More agricultural datasets.
- Expanded agricultural knowledge and recommendations.

## 11. Project Information

| Detail | Information |
|---|---|
| Project Name | AgriVision |
| Project Category | Artificial Intelligence and Machine Learning |
| Domain | Smart Agriculture |
| Frontend | React.js, Vite, Tailwind CSS |
| Backend | Python, FastAPI |
| Computer Vision Model | EfficientNet-B0 |
| NLP Model | DistilBERT |
| Semantic Retrieval | Sentence Transformers and FAISS |
| PDF Generation | ReportLab |
| Initial Crop Coverage | Wheat, Rice, Potato and Tomato |
| Deployment | Local Application |

