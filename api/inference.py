"""InferenceEngine — wraps Vocabulary, TargetedABSAModel, and asyncio.Lock.

Isolates all ML-specific logic from the FastAPI HTTP layer (api/main.py).
Run from project root with PYTHONPATH=. so that 'from vocabulary import Vocabulary'
and 'from model.targeted_absa import TargetedABSAModel' resolve correctly (D-03).
"""

import asyncio
import heapq
import os
import time
import unicodedata
from pathlib import Path

import torch
import torch.nn.functional as F

from vocabulary import Vocabulary
from model.targeted_absa import TargetedABSAModel

# ---------------------------------------------------------------------------
# Encoder key → HuggingFace model-ID mapping.
# Duplicated here (NOT imported from train.py) to avoid pulling in all
# training-only dependencies at API startup (D-03).
# ---------------------------------------------------------------------------
ENCODER_MAP: dict[str, str] = {
    "phobert_base": "vinai/phobert-base",
    "phobert_large": "vinai/phobert-large",
    "xlm_roberta_base": "xlm-roberta-base",
    "xlm_roberta_large": "xlm-roberta-large",
}

# Max token length used during inference — same as eval config (shorter than
# Vocabulary's default of 256 to keep latency lower on CPU).
_MAX_INFERENCE_LENGTH: int = 128


class InferenceEngine:
    """Loads the ABSA model at construction time and exposes predict().

    Attributes consumed by api/main.py:
        ready (bool)         — False until model loaded; /predict returns 503 while False
        lock (asyncio.Lock)  — Used as "async with engine.lock" to serialize requests
        encoder_key (str)    — Encoder key string, e.g. "phobert_large"
        encoder_name (str)   — Resolved HuggingFace ID, e.g. "vinai/phobert-large"
        attention_key (str)  — e.g. "target_conditioned_attention"
        label_structure (str)— e.g. "multitask_aspect_sentiment"
        device (torch.device)— Compute device used for inference
    """

    def __init__(self) -> None:
        # ------------------------------------------------------------------
        # Step 1 — Base attributes (set before anything can raise, so that
        #           callers can always inspect .ready and .lock safely).
        # ------------------------------------------------------------------
        self.ready: bool = False
        self.lock: asyncio.Lock = asyncio.Lock()

        # ------------------------------------------------------------------
        # Step 2 — Read env vars with defaults (D-01, D-05).
        # ------------------------------------------------------------------
        checkpoint_path: str = os.getenv("CHECKPOINT_PATH", "checkpoints/best_model.pt")
        data_dir: str = os.getenv("DATA_DIR", "Data")
        device_str: str = os.getenv("DEVICE", "")
        encoder_key: str = os.getenv("MODEL_ENCODER_KEY", "phobert_large")
        attention_key: str = os.getenv("MODEL_ATTENTION_KEY", "target_conditioned_attention")
        label_structure: str = os.getenv("MODEL_LABEL_STRUCTURE", "multitask_aspect_sentiment")

        # ------------------------------------------------------------------
        # Step 3 — Store for /models endpoint.
        # ------------------------------------------------------------------
        self.encoder_key: str = encoder_key
        self.attention_key: str = attention_key
        self.label_structure: str = label_structure

        # ------------------------------------------------------------------
        # Step 4 — Resolve compute device.
        # ------------------------------------------------------------------
        if device_str:
            self.device: torch.device = torch.device(device_str)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------------------------------------------------
        # Step 5 — Validate encoder key and resolve HuggingFace model name.
        # ------------------------------------------------------------------
        if encoder_key not in ENCODER_MAP:
            valid_keys = list(ENCODER_MAP.keys())
            raise ValueError(
                f"Unknown MODEL_ENCODER_KEY={encoder_key!r}. "
                f"Valid keys: {valid_keys}"
            )
        encoder_name: str = ENCODER_MAP[encoder_key]
        self.encoder_name: str = encoder_name

        # ------------------------------------------------------------------
        # Step 6 — Build Vocabulary from the same three JSONL files used
        #           during training (D-04), ensuring label maps match the
        #           trained checkpoint.
        # ------------------------------------------------------------------
        data_files = [
            os.path.join(data_dir, "train.jsonl"),
            os.path.join(data_dir, "dev.jsonl"),
            os.path.join(data_dir, "test.jsonl"),
        ]
        print(f"[InferenceEngine] Loading Vocabulary from {data_dir}/ ...")
        self.vocab: Vocabulary = Vocabulary(data_files, encoder_name)
        print(
            f"[InferenceEngine] Vocabulary ready: "
            f"{self.vocab.num_aspects} aspects, "
            f"{self.vocab.num_sentiments} sentiments"
        )

        # ------------------------------------------------------------------
        # Step 7 — Graceful checkpoint degradation (API-06).
        #           If the checkpoint file is missing, log a warning and
        #           return early.  self.ready stays False; the server
        #           starts successfully and /predict returns 503.
        # ------------------------------------------------------------------
        if not Path(checkpoint_path).exists():
            print(
                f"[InferenceEngine] WARNING: Checkpoint not found at {checkpoint_path!r}. "
                "Set CHECKPOINT_PATH env var or copy best_model.pt to that path. "
                "/predict returns 503 until checkpoint is available."
            )
            return

        # ------------------------------------------------------------------
        # Step 8 — Construct model architecture matching the saved checkpoint.
        # NOTE: num_labels is NOT passed — not required for
        #       "multitask_aspect_sentiment" (would raise if passed anyway).
        # ------------------------------------------------------------------
        print(f"[InferenceEngine] Loading {encoder_name} from {checkpoint_path} ...")
        model = TargetedABSAModel(
            encoder_name=encoder_name,
            num_aspects=self.vocab.num_aspects,
            num_sentiments=self.vocab.num_sentiments,
            label_structure_key=label_structure,
            attention_key=attention_key,
            dropout=0.2,                # matches training; model.eval() disables it
            classifier_hidden_size=256,  # matches config/base.yaml
        )

        # ------------------------------------------------------------------
        # Step 9 — Load state dict; strip DataParallel "module." prefix if
        #           the checkpoint was saved from a multi-GPU run (Pitfall 4).
        # weights_only=True prevents arbitrary code execution (T-02-01).
        # ------------------------------------------------------------------
        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

        # ------------------------------------------------------------------
        # Step 10 — Finalize: move to device, switch to eval mode.
        # model.eval() is CRITICAL — it disables dropout layers so that
        # predictions are deterministic at inference time (Pitfall 1).
        # ------------------------------------------------------------------
        model.to(self.device)
        model.eval()
        self.model: TargetedABSAModel = model
        self.ready = True
        print(f"[InferenceEngine] Model ready on {self.device}.")

    # -----------------------------------------------------------------------
    # predict — synchronous; called inside "async with engine.lock" in
    # api/main.py so that only one inference runs at a time (API-06).
    # -----------------------------------------------------------------------

    def predict(self, text: str, target: str) -> dict:
        """Run inference on a (text, target) pair.

        Returns a dict whose keys match PredictResponse field names exactly
        so callers can do PredictResponse(**engine.predict(text, target)).

        Args:
            text:   Full review text (Vietnamese).
            target: Target phrase — must be an exact substring of text.

        Returns:
            {
                "aspect":          str,          # top-1 aspect label
                "sentiment":       str,          # top-1 sentiment label
                "aspect_probs":    list[dict],   # top-5 {label, score} sorted desc
                "sentiment_probs": list[dict],   # all 3 {label, score} sorted desc
                "latency_ms":      int,
            }

        Raises:
            ValueError: for empty/whitespace text, empty/whitespace target,
                        or target not being a substring of text (API-05).
        """
        # ------------------------------------------------------------------
        # Step 1 — Unicode NFC normalization + BOM strip (API-07, T-02-04).
        #           Applied to BOTH inputs before any other processing.
        # ------------------------------------------------------------------
        text = unicodedata.normalize("NFC", text).replace("\ufeff", "")
        target = unicodedata.normalize("NFC", target).replace("\ufeff", "")

        # ------------------------------------------------------------------
        # Step 2 — Input validation (API-05).
        # ------------------------------------------------------------------
        if text.strip() == "":
            raise ValueError("text must not be empty or whitespace")
        if target.strip() == "":
            raise ValueError("target must not be empty or whitespace")
        if target not in text:
            raise ValueError("target must be an exact substring of text")

        # ------------------------------------------------------------------
        # Step 3 — Record wall-clock start time.
        # ------------------------------------------------------------------
        t0 = time.perf_counter()

        # ------------------------------------------------------------------
        # Step 4 — Tokenize at inference max_length (128 matches eval config).
        # ------------------------------------------------------------------
        encoded = self.vocab.tokenize_text_pair(
            text, target, max_length=_MAX_INFERENCE_LENGTH
        )

        # ------------------------------------------------------------------
        # Step 5 — Build batch tensors (unsqueeze(0) adds batch dimension).
        #           token_type_ids is BERT-style only; PhoBERT/XLM-R omit it.
        # ------------------------------------------------------------------
        input_ids = encoded["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = encoded["attention_mask"].unsqueeze(0).to(self.device)
        target_mask = encoded["target_mask"].unsqueeze(0).to(self.device)
        model_kwargs: dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": target_mask,
        }
        if "token_type_ids" in encoded:
            model_kwargs["token_type_ids"] = (
                encoded["token_type_ids"].unsqueeze(0).to(self.device)
            )

        # ------------------------------------------------------------------
        # Step 6 — Run model forward pass under no_grad to save memory and
        #           skip gradient computation.
        # ------------------------------------------------------------------
        with torch.no_grad():
            outputs = self.model(**model_kwargs)
        # outputs: {"aspect_logits": [1, num_aspects], "sentiment_logits": [1, num_sentiments], ...}

        # ------------------------------------------------------------------
        # Step 7 — Decode aspect predictions.
        #           Pitfall 7: 86 aspect classes — return top-5 only.
        # ------------------------------------------------------------------
        aspect_probs_t = F.softmax(outputs["aspect_logits"][0], dim=-1)
        top_aspect_id = int(aspect_probs_t.argmax().item())
        aspect_label: str = self.vocab.id2aspect[top_aspect_id]

        all_asp_pairs = [
            (self.vocab.id2aspect[i], round(float(p), 4))
            for i, p in enumerate(aspect_probs_t)
        ]
        top5_aspects = [
            {"label": lbl, "score": sc}
            for lbl, sc in heapq.nlargest(5, all_asp_pairs, key=lambda x: x[1])
        ]

        # ------------------------------------------------------------------
        # Step 8 — Decode sentiment predictions (all 3 classes returned).
        # ------------------------------------------------------------------
        sent_probs_t = F.softmax(outputs["sentiment_logits"][0], dim=-1)
        top_sent_id = int(sent_probs_t.argmax().item())
        sentiment_label: str = self.vocab.id2sentiment[top_sent_id]

        all_sent_pairs = [
            (self.vocab.id2sentiment[i], round(float(p), 4))
            for i, p in enumerate(sent_probs_t)
        ]
        sentiment_probs = [
            {"label": lbl, "score": sc}
            for lbl, sc in sorted(all_sent_pairs, key=lambda x: x[1], reverse=True)
        ]

        # ------------------------------------------------------------------
        # Step 9 — Compute latency and build return dict.
        #           Keys match PredictResponse field names exactly.
        # ------------------------------------------------------------------
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "aspect": aspect_label,
            "sentiment": sentiment_label,
            "aspect_probs": top5_aspects,
            "sentiment_probs": sentiment_probs,
            "latency_ms": latency_ms,
        }
