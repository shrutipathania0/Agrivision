"""
AgriVision NLP Engine Training & Semantic Indexing Pipeline
Components:
1. DistilBERT fine-tuning for 6 agricultural query intents (symptoms, causes, prevention, management, fertilizer, general)
2. SentenceTransformers (all-MiniLM-L6-v2) + FAISS (IndexFlatIP) dense semantic retrieval index
3. Grounded knowledge base ingestion (PAU Ludhiana, CCS HAU Hisar, ICAR-IIWBR, ICAR-CPRI)
4. Evaluation: Intent Macro-F1, Retrieval Recall@K, and MRR
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NLP_MODELS_DIR = Path("models/nlp/distilbert_intent_classifier").resolve()
EMBEDDINGS_DIR = Path("models/embeddings").resolve()
KB_DIR = Path("knowledge_base").resolve()
REPORTS_DIR = Path("reports/training").resolve()


class AgriculturalIntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_intent_classifier():
    print("=" * 70)
    print("TRAINING DISTILBERT INTENT CLASSIFIER")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    NLP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open("training/nlp/intent_dataset.json") as f:
        data = json.load(f)

    intents = data["intents"]
    intent_to_idx = {name: i for i, name in enumerate(intents)}
    idx_to_intent = {i: name for i, name in enumerate(intents)}

    texts = [item["text"] for item in data["dataset"]]
    labels = [intent_to_idx[item["intent"]] for item in data["dataset"]]

    # Stratified train/val split (80/20)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.20, random_state=SEED, stratify=labels
    )

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(intents)
    ).to(DEVICE)

    train_dataset = AgriculturalIntentDataset(train_texts, train_labels, tokenizer)
    val_dataset = AgriculturalIntentDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    epochs = 8

    best_val_macro_f1 = 0.0
    history = {"train_loss": [], "val_acc": [], "val_macro_f1": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            b_labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=b_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                b_labels = batch["labels"].to(DEVICE)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(b_labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        history["train_loss"].append(avg_train_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)

        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_acc:.3f} | Val Macro-F1: {val_macro_f1:.3f}")

        if val_macro_f1 >= best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            model.save_pretrained(NLP_MODELS_DIR)
            tokenizer.save_pretrained(NLP_MODELS_DIR)

    # Save intent label mapping
    mapping = {
        "intents": intents,
        "intent_to_idx": intent_to_idx,
        "idx_to_intent": idx_to_intent,
        "best_val_macro_f1": best_val_macro_f1
    }
    with open(NLP_MODELS_DIR / "intent_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"\n[OK] Intent Classifier saved to {NLP_MODELS_DIR} (Best Macro-F1: {best_val_macro_f1:.3f})")


def build_faiss_semantic_index():
    print("\n" + "=" * 70)
    print("BUILDING DENSE FAISS VECTOR RETRIEVAL INDEX")
    print("=" * 70)

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    passages = []

    # 1. Ingest crop disease records
    for crop_file in (KB_DIR / "diseases").glob("*.json"):
        with open(crop_file) as f:
            disease_records = json.load(f)
            for rec in disease_records:
                crop = rec["crop"]
                disease = rec["disease"]
                sources = rec.get("sources", [])

                # Symptom passage
                passages.append({
                    "crop": crop,
                    "disease": disease,
                    "intent": "disease_symptoms",
                    "text": f"Symptoms of {crop} {disease}: {rec['symptoms']}",
                    "sources": sources
                })

                # Causes passage
                passages.append({
                    "crop": crop,
                    "disease": disease,
                    "intent": "disease_causes",
                    "text": f"Causes and favorable weather for {crop} {disease} ({rec.get('pathogen', '')}): {rec.get('favorable_conditions', '')}. Regional status: {rec.get('regional_status', '')}",
                    "sources": sources
                })

                # Prevention passage
                prev_text = " ".join(rec.get("prevention_practices", []))
                passages.append({
                    "crop": crop,
                    "disease": disease,
                    "intent": "disease_prevention",
                    "text": f"Prevention practices for {crop} {disease}: {prev_text}",
                    "sources": sources
                })

                # Management passage
                mgmt_text = " ".join(rec.get("management_practices", []))
                safety = rec.get("safety_warning", "")
                passages.append({
                    "crop": crop,
                    "disease": disease,
                    "intent": "disease_management",
                    "text": f"Management and control practices for {crop} {disease}: {mgmt_text} Safety notice: {safety}",
                    "sources": sources
                })

    print(f"Extracted {len(passages)} structured knowledge passages from institutional records.")

    # 2. Encode passages with SentenceTransformer
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading dense embedding model '{embed_model_name}'...")
    embedder = SentenceTransformer(embed_model_name, device=DEVICE)

    texts = [p["text"] for p in passages]
    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

    # L2 normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Built FAISS IndexFlatIP with {index.ntotal} vectors of dimension {dim}.")

    # Save index & metadata
    faiss.write_index(index, str(EMBEDDINGS_DIR / "knowledge_faiss.index"))
    with open(EMBEDDINGS_DIR / "knowledge_passages.json", "w") as f:
        json.dump(passages, f, indent=2)

    # 3. Retrieval evaluation (Recall@K and MRR)
    print("\nRunning Retrieval Evaluation Benchmark...")
    test_queries = [
        {"q": "What are the early signs of potato late blight?", "target_crop": "Potato", "target_disease": "Late Blight"},
        {"q": "How to prevent yellow rust in wheat in Punjab?", "target_crop": "Wheat", "target_disease": "Yellow Rust"},
        {"q": "What causes blast lesions in basmati paddy?", "target_crop": "Rice", "target_disease": "Blast"},
        {"q": "How to manage early blight on tomato leaves?", "target_crop": "Tomato", "target_disease": "Early Blight"},
        {"q": "What are symptoms of bacterial leaf blight in rice?", "target_crop": "Rice", "target_disease": "Bacterial Leaf Blight"}
    ]

    recalls_at_1, recalls_at_3, recalls_at_5 = [], [], []
    rr_scores = []

    for tq in test_queries:
        q_emb = embedder.encode([tq["q"]], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        k = 5
        D, I = index.search(q_emb, k)
        retrieved_indices = I[0]

        hit_rank = None
        for rank, idx in enumerate(retrieved_indices, 1):
            cand = passages[idx]
            if cand["crop"].lower() == tq["target_crop"].lower() and cand["disease"].lower() == tq["target_disease"].lower():
                hit_rank = rank
                break

        recalls_at_1.append(1 if hit_rank == 1 else 0)
        recalls_at_3.append(1 if hit_rank and hit_rank <= 3 else 0)
        recalls_at_5.append(1 if hit_rank and hit_rank <= 5 else 0)
        rr_scores.append(1.0 / hit_rank if hit_rank else 0.0)

    print(f"-> Recall@1: {np.mean(recalls_at_1):.3f}")
    print(f"-> Recall@3: {np.mean(recalls_at_3):.3f}")
    print(f"-> Recall@5: {np.mean(recalls_at_5):.3f}")
    print(f"-> Mean Reciprocal Rank (MRR): {np.mean(rr_scores):.3f}")

    retrieval_metrics = {
        "num_passages": len(passages),
        "embedding_dimension": dim,
        "recall_at_1": round(float(np.mean(recalls_at_1)), 4),
        "recall_at_3": round(float(np.mean(recalls_at_3)), 4),
        "recall_at_5": round(float(np.mean(recalls_at_5)), 4),
        "mrr": round(float(np.mean(rr_scores)), 4)
    }
    with open(REPORTS_DIR / "nlp_retrieval_metrics.json", "w") as f:
        json.dump(retrieval_metrics, f, indent=2)

    print(f"[OK] Knowledge FAISS Index & passages exported to {EMBEDDINGS_DIR}")


if __name__ == "__main__":
    train_intent_classifier()
    build_faiss_semantic_index()
