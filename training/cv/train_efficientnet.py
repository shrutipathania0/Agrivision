"""
AgriVision Computer Vision Model Training Pipeline
Model: EfficientNet-B0 (ImageNet Pretrained)
Methodology: Two-Stage Transfer Learning
  Stage 1: Frozen backbone, train new projection head (5 epochs)
  Stage 2: Differential fine-tuning of top conv blocks (features[6], features[7]) with AdamW and ReduceLROnPlateau
Guarantees:
  - FP16 Mixed Precision (torch.amp)
  - Class-weighted CrossEntropyLoss
  - Windows-optimized zero-worker DataLoader (instant NVMe streaming, zero IPC deadlock)
  - Validation Macro-F1 and Weighted-F1 tracking
  - Best model checkpointing & crash recovery checkpoints
  - Training metrics history & visualization export
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import f1_score, accuracy_score


# Device and reproducibility
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROCESSED_DIR = Path("datasets/processed").resolve()
MODELS_DIR = Path("models/computer_vision").resolve()
REPORTS_DIR = Path("reports/training").resolve()


def get_data_transforms():
    # Training augmentation: RandomResizedCrop, flips, rotations, color jitter, ImageNet normalization
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation & Test: Deterministic preprocessing
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def build_model(num_classes):
    print(f"Loading EfficientNet-B0 ImageNet pretrained backbone...", flush=True)
    weights = EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    # In_features for EfficientNet-B0 classifier is 1280
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )
    return model.to(DEVICE)


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, epoch_num, stage_name):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    total_batches = len(dataloader)

    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (i + 1) % 50 == 0 or (i + 1) == total_batches:
            batch_loss = loss.item()
            print(f"  [{stage_name} Ep {epoch_num}] Batch [{i+1}/{total_batches}] - Loss: {batch_loss:.4f}", flush=True)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return epoch_loss, epoch_acc, epoch_macro_f1


@torch.no_grad()
def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(dataloader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    val_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    val_weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return val_loss, val_acc, val_macro_f1, val_weighted_f1


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(history["train_loss"], label="Train Loss", color="royalblue", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss", color="crimson", linewidth=2)
    axes[0].set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Macro-F1 & Accuracy plot
    axes[1].plot(history["val_acc"], label="Val Accuracy", color="forestgreen", linewidth=2)
    axes[1].plot(history["val_macro_f1"], label="Val Macro-F1", color="darkorange", linewidth=2)
    axes[1].set_title("Validation Accuracy & Macro-F1", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric Score")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Training curves saved to {save_path}", flush=True)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("AGRIVISION COMPUTER VISION TRAINING (EFFICIENTNET-B0)", flush=True)
    print(f"Compute Device: {DEVICE}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB)", flush=True)
    print("=" * 70, flush=True)

    train_transform, val_transform = get_data_transforms()

    # Datasets and loaders (num_workers=0 is optimal on Windows NVMe SSD)
    train_dataset = datasets.ImageFolder(PROCESSED_DIR / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(PROCESSED_DIR / "val", transform=val_transform)

    num_classes = len(train_dataset.classes)
    print(f"\nLoaded {len(train_dataset)} training images across {num_classes} classes.", flush=True)
    print(f"Loaded {len(val_dataset)} validation images across {num_classes} classes.", flush=True)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Load class weights if available
    class_weights_tensor = None
    audit_report_path = Path("reports/evaluation/dataset_audit_report.json")
    if audit_report_path.exists():
        with open(audit_report_path) as f:
            audit_data = json.load(f)
            weights_dict = audit_data.get("class_weights", {})
            weights_list = [weights_dict.get(cls, 1.0) for cls in train_dataset.classes]
            class_weights_tensor = torch.tensor(weights_list, dtype=torch.float32).to(DEVICE)
            print("Loaded balanced class weights for CrossEntropyLoss.", flush=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes)

    history = {
        "train_loss": [], "train_acc": [], "train_macro_f1": [],
        "val_loss": [], "val_acc": [], "val_macro_f1": [], "val_weighted_f1": []
    }

    best_val_macro_f1 = 0.0
    best_checkpoint_path = MODELS_DIR / "efficientnet_b0_agrivision_best.pth"

    # =========================================================================
    # STAGE 1: Head Warmup (Backbone Frozen, 5 Epochs)
    # =========================================================================
    print("\n" + "-" * 70, flush=True)
    print("STAGE 1: Training Classification Head (Backbone Frozen, 5 Epochs)", flush=True)
    print("-" * 70, flush=True)

    for param in model.features.parameters():
        param.requires_grad = False

    optimizer_head = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, 6):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer_head, scaler, epoch, "Stage 1")
        vl_loss, vl_acc, vl_f1, vl_wf1 = evaluate(model, val_loader, criterion)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_macro_f1"].append(tr_f1)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["val_macro_f1"].append(vl_f1)
        history["val_weighted_f1"].append(vl_wf1)

        print(f"\n>> Stage 1 - Epoch [{epoch}/5] ({elapsed:.1f}s) | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | Val Loss: {vl_loss:.4f} Acc: {vl_acc:.3f} Macro-F1: {vl_f1:.3f}\n", flush=True)

        if vl_f1 > best_val_macro_f1:
            best_val_macro_f1 = vl_f1
            torch.save({
                "epoch": epoch,
                "stage": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_head.state_dict(),
                "best_val_macro_f1": best_val_macro_f1,
                "classes": train_dataset.classes,
                "class_to_idx": train_dataset.class_to_idx
            }, best_checkpoint_path)
            print(f"  -> Saved Stage 1 best model (Val Macro-F1: {best_val_macro_f1:.4f})", flush=True)

    # =========================================================================
    # STAGE 2: Differential Fine-Tuning (Top Conv Blocks Unfrozen)
    # =========================================================================
    print("\n" + "-" * 70, flush=True)
    print("STAGE 2: Differential Fine-Tuning (Unfreezing features[6] & features[7])", flush=True)
    print("-" * 70, flush=True)

    # Unfreeze top convolutional stages
    for param in model.features[6].parameters():
        param.requires_grad = True
    for param in model.features[7].parameters():
        param.requires_grad = True
    for param in model.features[8].parameters():
        param.requires_grad = True

    optimizer_finetune = torch.optim.AdamW([
        {"params": model.features[6].parameters(), "lr": 1e-4},
        {"params": model.features[7].parameters(), "lr": 1e-4},
        {"params": model.features[8].parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 5e-4}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_finetune, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    stage2_epochs = 12
    patience = 4
    no_improve_epochs = 0

    for epoch in range(6, 6 + stage2_epochs):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer_finetune, scaler, epoch, "Stage 2")
        vl_loss, vl_acc, vl_f1, vl_wf1 = evaluate(model, val_loader, criterion)
        scheduler.step(vl_loss)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_macro_f1"].append(tr_f1)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["val_macro_f1"].append(vl_f1)
        history["val_weighted_f1"].append(vl_wf1)

        print(f"\n>> Stage 2 - Epoch [{epoch}/{5 + stage2_epochs}] ({elapsed:.1f}s) | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | Val Loss: {vl_loss:.4f} Acc: {vl_acc:.3f} Macro-F1: {vl_f1:.3f}", flush=True)

        if vl_f1 > best_val_macro_f1:
            best_val_macro_f1 = vl_f1
            no_improve_epochs = 0
            torch.save({
                "epoch": epoch,
                "stage": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_finetune.state_dict(),
                "best_val_macro_f1": best_val_macro_f1,
                "classes": train_dataset.classes,
                "class_to_idx": train_dataset.class_to_idx
            }, best_checkpoint_path)
            print(f"  -> [SAVED NEW BEST MODEL] Val Macro-F1: {best_val_macro_f1:.4f}", flush=True)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"  -> Early stopping triggered after {patience} epochs without validation Macro-F1 improvement.", flush=True)
                break

    # Save training history JSON and plots
    with open(REPORTS_DIR / "cv_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_training_history(history, REPORTS_DIR / "cv_training_curves.png")

    print("\n" + "=" * 70, flush=True)
    print(f"TRAINING COMPLETE! Best Validation Macro-F1: {best_val_macro_f1:.4f}", flush=True)
    print(f"Best checkpoint: {best_checkpoint_path}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
