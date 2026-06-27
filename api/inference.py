import asyncio
import os
import time
import unicodedata
from pathlib import Path

import torch
import torch.nn.functional as F

from vocabulary import Vocabulary
from model.targeted_absa import TargetedABSAModel

ENCODER_MAP = {
    "phobert_base": "vinai/phobert-base",
    "phobert_large": "vinai/phobert-large",
    "xlm_roberta_base": "xlm-roberta-base",
    "xlm_roberta_large": "xlm-roberta-large",
}

_MAX_LENGTH = 128


class InferenceEngine:
    def __init__(self) -> None:
        self.ready = False
        self.lock = asyncio.Lock()

        checkpoint_path = os.getenv("CHECKPOINT_PATH", "checkpoints/best_model.pt")
        data_dir = os.getenv("DATA_DIR", "Data")
        device_str = os.getenv("DEVICE", "")
        encoder_key = os.getenv("MODEL_ENCODER_KEY", "phobert_large")
        attention_key = os.getenv("MODEL_ATTENTION_KEY", "target_conditioned_attention")
        label_structure = os.getenv("MODEL_LABEL_STRUCTURE", "multitask_with_joint_aux")

        self.encoder_key = encoder_key
        self.attention_key = attention_key
        self.label_structure = label_structure
        self.device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if encoder_key not in ENCODER_MAP:
            raise ValueError(f"Unknown MODEL_ENCODER_KEY={encoder_key!r}. Valid: {list(ENCODER_MAP)}")
        encoder_name = ENCODER_MAP[encoder_key]
        self.encoder_name = encoder_name

        print(f"[InferenceEngine] Loading Vocabulary from {data_dir}/ ...")
        self.vocab = Vocabulary([os.path.join(data_dir, "train.jsonl")], encoder_name)
        print(f"[InferenceEngine] {self.vocab.num_aspects} aspects, {self.vocab.num_sentiments} sentiments")

        if not Path(checkpoint_path).exists():
            print(f"[InferenceEngine] WARNING: checkpoint not found at {checkpoint_path!r}")
            return

        print(f"[InferenceEngine] Loading {encoder_name} from {checkpoint_path} ...")
        model = TargetedABSAModel(
            encoder_name=encoder_name,
            num_aspects=self.vocab.num_aspects,
            num_sentiments=self.vocab.num_sentiments,
            num_labels=self.vocab.num_labels,
            label_structure_key=label_structure,
            attention_key=attention_key,
            dropout=0.2,
            classifier_hidden_size=256,
        )

        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("joint_head")}
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()

        self.model = model
        self.ready = True
        print(f"[InferenceEngine] Ready on {self.device}.")

    def predict(self, text: str, target: str) -> dict:
        text = unicodedata.normalize("NFC", text).replace("\ufeff", "")
        target = unicodedata.normalize("NFC", target).replace("\ufeff", "")

        if not text.strip():
            raise ValueError("text must not be empty")
        if not target.strip():
            raise ValueError("target must not be empty")
        if target not in text:
            raise ValueError("target must be an exact substring of text")

        t0 = time.perf_counter()

        encoded = self.vocab.tokenize_text_pair(text, target, max_length=_MAX_LENGTH)
        input_ids = encoded["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = encoded["attention_mask"].unsqueeze(0).to(self.device)
        target_mask = encoded["target_mask"].unsqueeze(0).to(self.device)
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "target_mask": target_mask}
        if "token_type_ids" in encoded:
            kwargs["token_type_ids"] = encoded["token_type_ids"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(**kwargs)

        aspect_probs = F.softmax(outputs["aspect_logits"][0], dim=-1)
        top_aspect = self.vocab.id2aspect[int(aspect_probs.argmax())]
        top5_aspects = sorted(
            [{"label": self.vocab.id2aspect[i], "score": round(float(p), 4)} for i, p in enumerate(aspect_probs)],
            key=lambda x: x["score"], reverse=True
        )[:5]

        sent_probs = F.softmax(outputs["sentiment_logits"][0], dim=-1)
        top_sentiment = self.vocab.id2sentiment[int(sent_probs.argmax())]
        all_sentiments = sorted(
            [{"label": self.vocab.id2sentiment[i], "score": round(float(p), 4)} for i, p in enumerate(sent_probs)],
            key=lambda x: x["score"], reverse=True
        )

        return {
            "aspect": top_aspect,
            "sentiment": top_sentiment,
            "aspect_probs": top5_aspects,
            "sentiment_probs": all_sentiments,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }
