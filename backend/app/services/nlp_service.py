"""
AgriVision NLP Advisory Service
Components:
- DistilBERT intent classification
- Dense semantic retrieval via FAISS (IndexFlatIP) and SentenceTransformers
- Scan context injection for multi-turn advisory continuity
- Grounded response construction citing PAU, CCS HAU, and ICAR
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

from backend.app.core.config import settings


class NLPAdvisoryService:
    def __init__(self):
        self.device = torch.device(settings.DEVICE)
        self.tokenizer = None
        self.intent_model = None
        self.idx_to_intent = {}
        self.embedder = None
        self.faiss_index = None
        self.passages = []
        self.knowledge_base = {}
        self.load_resources()

    def load_resources(self):
        # 1. Load Knowledge Base files
        kb_dir = settings.KNOWLEDGE_BASE_DIR / "diseases"
        if kb_dir.exists():
            for f in kb_dir.glob("*.json"):
                with open(f) as fp:
                    records = json.load(fp)
                    for r in records:
                        key = f"{r['crop'].lower()}___{r['disease'].lower().replace(' ', '_')}"
                        self.knowledge_base[key] = r
            print(f"[NLPService] Loaded {len(self.knowledge_base)} disease advisory records.")

        # 2. Load DistilBERT Intent Classifier
        nlp_dir = settings.NLP_MODEL_DIR
        if (nlp_dir / "config.json").exists():
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(nlp_dir)
                self.intent_model = AutoModelForSequenceClassification.from_pretrained(nlp_dir).to(self.device)
                self.intent_model.eval()
                with open(nlp_dir / "intent_mapping.json") as fp:
                    data = json.load(fp)
                    self.idx_to_intent = {int(k): v for k, v in data["idx_to_intent"].items()}
                print("[NLPService] DistilBERT Intent Classifier loaded successfully.")
            except Exception as e:
                print(f"[NLPService Error] Failed to load DistilBERT: {e}")

        # 3. Load FAISS Semantic Retrieval Index
        emb_dir = settings.NLP_EMBEDDINGS_DIR
        index_path = emb_dir / "knowledge_faiss.index"
        passages_path = emb_dir / "knowledge_passages.json"

        if index_path.exists() and passages_path.exists():
            try:
                self.faiss_index = faiss.read_index(str(index_path))
                with open(passages_path) as fp:
                    self.passages = json.load(fp)
                self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=self.device)
                print(f"[NLPService] FAISS Index ({self.faiss_index.ntotal} vectors) and Embedder loaded.")
            except Exception as e:
                print(f"[NLPService Error] Failed to load FAISS index: {e}")

    def predict_intent(self, text: str) -> str:
        """Classify user question intent using fine-tuned DistilBERT."""
        if not self.intent_model or not self.tokenizer:
            return "general_farming_inquiry"

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.intent_model(**inputs)
            pred_idx = torch.argmax(outputs.logits, dim=1).item()
            return self.idx_to_intent.get(pred_idx, "general_farming_inquiry")

    def answer_question(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a grounded, cited answer using scan context and semantic retrieval."""
        intent = self.predict_intent(question)

        # Check if question relates to active scan context
        scan_crop = context.get("crop") if context else None
        scan_disease = context.get("disease") if context else None

        if scan_crop and scan_disease:
            kb_key = f"{scan_crop.lower()}___{scan_disease.lower().replace(' ', '_')}"
            rec = self.knowledge_base.get(kb_key)
            if rec:
                # Direct contextual match based on classified intent
                return self._synthesize_contextual_answer(question, intent, rec, context)

        # Fallback to FAISS dense semantic retrieval
        if self.faiss_index and self.embedder and self.passages:
            return self._retrieve_semantic_answer(question, intent, scan_crop, scan_disease)

        return {
            "question": question,
            "intent": intent,
            "answer": "AgriVision advisory services are currently operating in offline mode. Please refer to Punjab Agricultural University (PAU) or local Krishi Vigyan Kendra advisories.",
            "sources": [{"institution": "PAU Ludhiana", "url": "https://www.pau.edu"}],
            "context_used": context
        }

    def _synthesize_contextual_answer(self, question: str, intent: str, rec: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        crop = rec["crop"]
        disease = rec["disease"]
        sources = rec.get("sources", [])
        safety = rec.get("safety_warning", "Consult local KVK for verified agricultural safety procedures.")

        if intent == "disease_symptoms":
            answer = f"According to verified institutional records for {crop} ({disease}):\n\nSymptoms: {rec.get('symptoms', '')}\n\nFavorable Conditions: {rec.get('favorable_conditions', '')}"
        elif intent == "disease_causes":
            answer = f"Etiology for {crop} {disease}:\n\nPathogen: {rec.get('pathogen', 'N/A')}\nTriggers: {rec.get('favorable_conditions', '')}\nRegional Notes: {rec.get('regional_status', '')}"
        elif intent == "disease_prevention":
            practices = "\n".join([f"• {p}" for p in rec.get("prevention_practices", [])])
            answer = f"Recommended preventive practices for {crop} ({disease}) by PAU/ICAR:\n\n{practices}"
        elif intent == "disease_management":
            practices = "\n".join([f"• {p}" for p in rec.get("management_practices", [])])
            answer = f"Integrated disease management for {crop} ({disease}):\n\n{practices}\n\n⚠️ Safety Notice: {safety}"
        else:
            # Combined overview
            practices = "\n".join([f"• {p}" for p in rec.get("management_practices", [])[:2]])
            answer = f"For {crop} {disease} ({rec.get('pathogen', '')}):\n\n{rec.get('symptoms', '')}\n\nKey Management Steps:\n{practices}\n\n⚠️ Safety Notice: {safety}"

        return {
            "question": question,
            "intent": intent,
            "answer": answer,
            "sources": sources,
            "context_used": context
        }

    def _retrieve_semantic_answer(self, question: str, intent: str, filter_crop: Optional[str] = None, filter_disease: Optional[str] = None) -> Dict[str, Any]:
        q_emb = self.embedder.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)

        D, I = self.faiss_index.search(q_emb, k=5)
        top_indices = I[0]

        selected_passage = None
        for idx in top_indices:
            cand = self.passages[idx]
            if filter_crop and cand["crop"].lower() != filter_crop.lower():
                continue
            selected_passage = cand
            break

        if not selected_passage and len(top_indices) > 0:
            selected_passage = self.passages[top_indices[0]]

        answer = selected_passage["text"] if selected_passage else "No matching record found in knowledge base."
        sources = selected_passage.get("sources", [{"institution": "ICAR / PAU", "url": "https://www.pau.edu"}]) if selected_passage else []

        return {
            "question": question,
            "intent": intent,
            "answer": f"Grounded Advisory ({selected_passage.get('crop', '')} {selected_passage.get('disease', '')}):\n\n{answer}",
            "sources": sources,
            "context_used": {"crop": filter_crop, "disease": filter_disease} if filter_crop else None
        }

    def get_advisory_for_class(self, crop: str, disease: str) -> Optional[Dict[str, Any]]:
        """Fetch full static advisory record for diagnostic cards."""
        key = f"{crop.lower()}___{disease.lower().replace(' ', '_')}"
        return self.knowledge_base.get(key)


nlp_service = NLPAdvisoryService()
