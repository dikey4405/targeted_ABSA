import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import yaml
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from dataloader import build_dataloader
from model.targeted_absa import TargetedABSAModel
from vocabulary import Vocabulary

transformers.logging.set_verbosity_error()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def resolve_path(path: str) -> Path:
    raw = Path(path)
    candidates = [
        raw,
        PROJECT_ROOT / raw,
        WORKSPACE_ROOT / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (WORKSPACE_ROOT / raw).resolve()


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def inverse_frequency_weights(ids: Iterable[int], num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float)
    for idx in ids:
        counts[int(idx)] += 1.0

    weights = counts.sum() / counts.clamp(min=1.0)
    weights = weights / weights.mean().clamp(min=1e-6)
    weights[counts == 0] = 0.0
    return weights


class FocalLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


class ABSATrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        loss_key: str,
        lr: float,
        epochs: int,
        weight_decay: float,
        warmup_ratio: float,
        max_grad_norm: float,
        save_dir: Path,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.loss_key = loss_key
        self.max_grad_norm = max_grad_norm
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

        total_steps = max(1, len(train_loader) * epochs)
        warmup_steps = int(warmup_ratio * total_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.aspect_weights, self.sentiment_weights, self.label_weights = self._build_class_weights()

    def _build_class_weights(self):
        dataset = self.train_loader.dataset
        aspect_ids = [dataset.vocab.encode_aspect(s["aspect"]) for s in dataset.samples]
        sentiment_ids = [dataset.vocab.encode_sentiment(s["sentiment"]) for s in dataset.samples]
        label_ids = [dataset.vocab.encode_label(s["label"]) for s in dataset.samples]

        aspect_weights = inverse_frequency_weights(aspect_ids, dataset.vocab.num_aspects).to(self.device)
        sentiment_weights = inverse_frequency_weights(sentiment_ids, dataset.vocab.num_sentiments).to(self.device)
        label_weights = inverse_frequency_weights(label_ids, dataset.vocab.num_labels).to(self.device)
        return aspect_weights, sentiment_weights, label_weights

    def _ce(self, logits, targets, weight=None):
        return F.cross_entropy(logits, targets.to(self.device), weight=weight)

    def _focal(self, logits, targets, weight=None):
        return FocalLoss(weight=weight, gamma=2.0).to(self.device)(logits, targets.to(self.device))

    def compute_loss(self, outputs, batch):
        aspect_targets = batch["aspect_id"].to(self.device)
        sentiment_targets = batch["sentiment_id"].to(self.device)
        label_targets = batch["label_id"].to(self.device)

        if self.loss_key == "multitask_ce":
            loss_a = self._ce(outputs["aspect_logits"], aspect_targets)
            loss_s = self._ce(outputs["sentiment_logits"], sentiment_targets)
            return loss_a + loss_s

        if self.loss_key == "multitask_weighted_ce":
            loss_a = self._ce(outputs["aspect_logits"], aspect_targets, self.aspect_weights)
            loss_s = self._ce(outputs["sentiment_logits"], sentiment_targets, self.sentiment_weights)
            return loss_a + loss_s

        if self.loss_key == "focal_multitask":
            loss_a = self._focal(outputs["aspect_logits"], aspect_targets, self.aspect_weights)
            loss_s = self._focal(outputs["sentiment_logits"], sentiment_targets, self.sentiment_weights)
            return loss_a + loss_s

        if self.loss_key == "joint_ce":
            return self._ce(outputs["label_logits"], label_targets, self.label_weights)

        if self.loss_key == "joint_aux_loss":
            loss_a = self._ce(outputs["aspect_logits"], aspect_targets, self.aspect_weights)
            loss_s = self._ce(outputs["sentiment_logits"], sentiment_targets, self.sentiment_weights)
            loss_j = self._ce(outputs["label_logits"], label_targets, self.label_weights)
            return loss_a + loss_s + 0.5 * loss_j

        raise ValueError(f"Unsupported loss_key: {self.loss_key}")

    def _forward_pass(self, batch):
        model_inputs = {
            "input_ids": batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device),
            "target_mask": batch.get("target_mask", None),
            "aspect_ids": batch.get("aspect_id", None),
        }

        if model_inputs["target_mask"] is not None:
            model_inputs["target_mask"] = model_inputs["target_mask"].to(self.device)
        if model_inputs["aspect_ids"] is not None:
            model_inputs["aspect_ids"] = model_inputs["aspect_ids"].to(self.device)
        if "token_type_ids" in batch:
            model_inputs["token_type_ids"] = batch["token_type_ids"].to(self.device)

        return self.model(**model_inputs)

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]", leave=False)
        for batch in progress_bar:
            self.optimizer.zero_grad()
            outputs = self._forward_pass(batch)
            loss = self.compute_loss(outputs, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(1, len(self.train_loader))

    def evaluate(self):
        self.model.eval()
        val_loss = 0.0
        aspect_preds, aspect_targets = [], []
        sentiment_preds, sentiment_targets = [], []
        joint_preds, joint_targets = [], []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
                outputs = self._forward_pass(batch)
                val_loss += self.compute_loss(outputs, batch).item()

                if "aspect_logits" in outputs and "sentiment_logits" in outputs:
                    a_pred = torch.argmax(outputs["aspect_logits"], dim=-1).cpu().tolist()
                    s_pred = torch.argmax(outputs["sentiment_logits"], dim=-1).cpu().tolist()
                    a_true = batch["aspect_id"].cpu().tolist()
                    s_true = batch["sentiment_id"].cpu().tolist()

                    aspect_preds.extend(a_pred)
                    sentiment_preds.extend(s_pred)
                    aspect_targets.extend(a_true)
                    sentiment_targets.extend(s_true)
                    joint_preds.extend([f"{a}#{s}" for a, s in zip(a_pred, s_pred)])
                    joint_targets.extend([f"{a}#{s}" for a, s in zip(a_true, s_true)])

                elif "label_logits" in outputs:
                    pred = torch.argmax(outputs["label_logits"], dim=-1).cpu().tolist()
                    true = batch["label_id"].cpu().tolist()
                    joint_preds.extend(pred)
                    joint_targets.extend(true)

        metrics = {"val_loss": val_loss / max(1, len(self.val_loader))}
        if aspect_targets:
            metrics["aspect_macro_f1"] = f1_score(aspect_targets, aspect_preds, average="macro", zero_division=0)
            metrics["sentiment_macro_f1"] = f1_score(sentiment_targets, sentiment_preds, average="macro", zero_division=0)
            metrics["joint_macro_f1"] = f1_score(joint_targets, joint_preds, average="macro", zero_division=0)
            metrics["macro_f1"] = metrics["joint_macro_f1"]
        else:
            metrics["joint_macro_f1"] = f1_score(joint_targets, joint_preds, average="macro", zero_division=0)
            metrics["macro_f1"] = metrics["joint_macro_f1"]
        return metrics

    def train(self):
        best_f1 = -1.0
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            metrics = self.evaluate()

            print(
                f"   Epoch {epoch} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {metrics['val_loss']:.4f} | Joint Macro-F1: {metrics['macro_f1']:.4f}"
            )

            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                torch.save(self.model.state_dict(), self.save_dir / "best_model.pt")

        print(f"Hoan thanh. Best Joint Macro-F1: {best_f1:.4f}. Saved: {self.save_dir / 'best_model.pt'}\n")


def main():
    experiments_path = PROJECT_ROOT / "config" / "experiments.yaml"
    experiments_cfg = load_yaml(experiments_path)
    base_cfg = load_yaml(resolve_path(experiments_cfg["defaults"]["base"]))
    encoders_cfg = load_yaml(resolve_path(experiments_cfg["defaults"]["encoders"]))

    train_file = resolve_path(base_cfg["paths"]["train_file"])
    dev_file = resolve_path(base_cfg["paths"]["dev_file"])
    checkpoint_root = resolve_path(base_cfg["paths"]["checkpoint_dir"])

    data_cfg = base_cfg["data"]
    train_cfg = base_cfg["training"]
    model_cfg = base_cfg["model"]

    batch_size = int(data_cfg.get("train_batch_size", 16))
    eval_batch_size = int(data_cfg.get("eval_batch_size", batch_size))
    max_length = int(data_cfg.get("max_length", 256))
    num_workers = int(data_cfg.get("num_workers", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not dev_file.exists():
        raise FileNotFoundError(f"Dev file not found: {dev_file}")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Train file: {train_file}")
    print(f"Dev file: {dev_file}")
    print(f"Checkpoint root: {checkpoint_root}")
    print(f"Device: {device}")

    experiment_groups = experiments_cfg.get("experiment_groups", {})
    total_runs = sum(len(group.get("runs", [])) for group in experiment_groups.values())
    print(f"Found {len(experiment_groups)} experiment group(s), {total_runs} run(s).\n")

    encoders = encoders_cfg["encoders"]

    for group_name, group_data in experiment_groups.items():
        print("=" * 70)
        print(f"GROUP: {group_name}")
        print(group_data.get("description", ""))
        print("=" * 70)

        for run in group_data.get("runs", []):
            run_name = run["name"]
            encoder_key = run.get("encoder_key", model_cfg.get("encoder_key", "phobert_base"))
            label_key = run.get("label_structure_key", model_cfg.get("label_structure_key", "multitask_aspect_sentiment"))
            attention_key = run.get("attention_key", model_cfg.get("attention_key", "cls_pooling"))
            loss_key = run.get("loss_key", model_cfg.get("loss_key", "multitask_ce"))

            encoder_info = encoders[encoder_key]
            encoder_name = encoder_info["model_name"]
            tokenizer_name = encoder_info.get("tokenizer_name", encoder_name)

            print(f"\nRUN: {run_name}")
            print(f"   Encoder: {encoder_name}")
            print(f"   Tokenizer: {tokenizer_name}")
            print(f"   Label structure: {label_key}")
            print(f"   Attention: {attention_key}")
            print(f"   Loss: {loss_key}")

            model = None
            trainer = None
            try:
                vocab = Vocabulary(data_path=[train_file, dev_file], model_name=tokenizer_name)
                train_loader = build_dataloader(
                    train_file,
                    vocab,
                    batch_size=batch_size,
                    max_length=max_length,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available(),
                )
                val_loader = build_dataloader(
                    dev_file,
                    vocab,
                    batch_size=eval_batch_size,
                    max_length=max_length,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available(),
                )

                model = TargetedABSAModel(
                    encoder_name=encoder_name,
                    num_aspects=vocab.num_aspects,
                    num_sentiments=vocab.num_sentiments,
                    num_labels=vocab.num_labels,
                    label_structure_key=label_key,
                    attention_key=attention_key,
                    dropout=float(model_cfg.get("dropout", 0.2)),
                    classifier_hidden_size=int(model_cfg.get("classifier_hidden_size", 256)),
                )

                trainer = ABSATrainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    loss_key=loss_key,
                    lr=float(train_cfg.get("learning_rate", 2.0e-5)),
                    epochs=int(train_cfg.get("epochs", 10)),
                    weight_decay=float(train_cfg.get("weight_decay", 0.01)),
                    warmup_ratio=float(train_cfg.get("warmup_ratio", 0.1)),
                    max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
                    save_dir=checkpoint_root / group_name / run_name,
                )
                trainer.train()

            except Exception as exc:
                print(f"ERROR in run '{run_name}': {exc}")
                print("Skipping this run and continuing...")

            finally:
                del trainer
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print("\nAll configured experiments finished.")


if __name__ == "__main__":
    main()
