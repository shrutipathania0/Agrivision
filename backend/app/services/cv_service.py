"""
AgriVision Computer Vision Inference Service
Responsibilities:
- Model loading (EfficientNet-B0) on CUDA/CPU
- Strict image validation & preprocessing
- Softmax confidence calibration & Shannon entropy uncertainty filtering
- Grad-CAM visual attention heatmap generation
"""

import io
import json
import base64
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models

from backend.app.core.config import settings


class ComputerVisionService:
    def __init__(self):
        self.device = torch.device(settings.DEVICE)
        self.model = None
        self.classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.gradients = None
        self.activations = None
        self.transform = transforms.Compose([
            transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.load_model()

    def load_model(self):
        # 1. Load class mapping
        mapping_path = settings.CV_CLASS_MAPPING_PATH
        if mapping_path.exists():
            with open(mapping_path) as f:
                data = json.load(f)
                self.classes = data["classes"]
                self.class_to_idx = data["class_to_idx"]
                self.idx_to_class = {int(k): v for k, v in data["idx_to_class"].items()}
        else:
            print(f"[WARN] Class mapping not found at {mapping_path}")
            return

        num_classes = len(self.classes)
        print(f"[CVService] Initializing EfficientNet-B0 with {num_classes} classes on {self.device}...")

        # 2. Build model architecture matching trained weights
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

        # 3. Load checkpoint
        ckpt_path = settings.CV_MODEL_PATH
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            print(f"[CVService] Successfully loaded weights from {ckpt_path}")
        else:
            print(f"[WARN] Trained checkpoint not found at {ckpt_path}. Model will run with initialized weights.")

        model.to(self.device)
        model.eval()
        self.model = model
        self._register_gradcam_hooks()

    def _register_gradcam_hooks(self):
        """Register forward and backward hooks on the last convolutional layer."""
        if not self.model:
            return

        target_layer = self.model.features[-1]

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def validate_image(self, file_bytes: bytes) -> Image.Image:
        """Validate format, readability, and dimensions."""
        if len(file_bytes) > 15 * 1024 * 1024:
            raise ValueError("File exceeds maximum size limit of 15MB.")

        try:
            image = Image.open(io.BytesIO(file_bytes))
            image.verify()
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError("Malformed or unreadable image file. Please upload a valid JPG/PNG.")

        if image.width < 64 or image.height < 64:
            raise ValueError("Image dimensions are too small. Minimum resolution is 64x64 pixels.")

        return image

    def predict(self, image: Image.Image, generate_gradcam: bool = True):
        if not self.model:
            raise RuntimeError("Computer Vision model is not initialized.")

        # Preprocessing
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True

        self.model.zero_grad()
        output = self.model(input_tensor)
        probabilities = torch.softmax(output, dim=1).squeeze(0)

        # Top-1 prediction
        top1_idx = torch.argmax(probabilities).item()
        confidence = probabilities[top1_idx].item()
        raw_class_name = self.idx_to_class.get(top1_idx, "Unknown")

        # Parse crop and disease name
        if "Non_Target" in raw_class_name:
            crop_name = "Non-Target / Unknown"
            disease_name = "Unsupported Crop or Non-Leaf Object"
            is_healthy = False
            is_ambiguous = True
        elif "___" in raw_class_name:
            crop_name, disease_name = raw_class_name.split("___", 1)
            disease_name = disease_name.replace("_", " ")
            is_healthy = bool("healthy" in disease_name.lower())
            is_ambiguous = False
        else:
            crop_name = "Crop"
            disease_name = raw_class_name.replace("_", " ")
            is_healthy = bool("healthy" in disease_name.lower())
            is_ambiguous = False

        # Shannon Entropy check for uncertainty
        probs_np = probabilities.detach().cpu().numpy()
        entropy = float(-np.sum(probs_np * np.log(probs_np + 1e-12)))
        if not is_ambiguous:
            is_ambiguous = bool(confidence < settings.CONFIDENCE_THRESHOLD or entropy > settings.ENTROPY_THRESHOLD)

        # Top alternatives
        topk = torch.topk(probabilities, k=min(4, len(self.classes)))
        alternatives = []
        for prob, idx in zip(topk.values.tolist(), topk.indices.tolist()):
            cls_name = self.idx_to_class.get(idx, f"Class {idx}")
            alternatives.append({
                "class_name": cls_name.replace("___", " - ").replace("_", " "),
                "confidence": round(prob, 4)
            })

        # Generate Grad-CAM saliency map
        gradcam_base64 = None
        if generate_gradcam and self.activations is not None:
            gradcam_base64 = self._generate_gradcam(output, top1_idx, image)

        return {
            "crop": crop_name,
            "disease": disease_name,
            "raw_class": raw_class_name,
            "confidence": round(confidence, 4),
            "entropy": round(float(entropy), 4),
            "is_healthy": is_healthy,
            "is_ambiguous": is_ambiguous,
            "top_alternatives": alternatives,
            "gradcam_base64": gradcam_base64
        }

    def _generate_gradcam(self, model_output, class_idx, original_image: Image.Image) -> str:
        """Compute Grad-CAM activation map and blend onto original image."""
        try:
            score = model_output[0, class_idx]
            self.model.zero_grad()
            score.backward(retain_graph=True)

            gradients = self.gradients.detach().cpu().numpy()[0]  # shape: (C, H, W)
            activations = self.activations.detach().cpu().numpy()[0]  # shape: (C, H, W)

            # Global average pooling of gradients
            weights = np.mean(gradients, axis=(1, 2))  # shape: (C,)
            cam = np.zeros(activations.shape[1:], dtype=np.float32)  # shape: (H, W)

            for i, w in enumerate(weights):
                cam += w * activations[i]

            cam = np.maximum(cam, 0)  # ReLU
            if np.max(cam) > 0:
                cam = cam / np.max(cam)  # Normalize to [0, 1]

            # Resize CAM to match original image dimensions
            orig_w, orig_h = original_image.size
            cam_resized = cv2.resize(cam, (orig_w, orig_h))
            heatmap = np.uint8(255 * cam_resized)
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Convert to RGB JPEG base64
            colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode(".jpg", colored_heatmap_rgb)
            return base64.b64encode(buffer).decode("utf-8")
        except Exception as e:
            print(f"[Grad-CAM Error] Failed to generate saliency map: {e}")
            return None


cv_service = ComputerVisionService()
