import argparse
import os
import random
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import transformers
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

transformers.logging.set_verbosity_error()

from vocabulary import Vocabulary
from dataloader import build_dataloader
from joint_decoding import build_label_indices, decode_structured_joint, structured_joint_logits
from model.targeted_absa import TargetedABSAModel

# ==========================================
# DICTIONARY MAPPING CÁC KEY TỪ YAML
# ==========================================
ENCODER_MAP = {
    "phobert_base": "vinai/phobert-base",
    "phobert_large": "vinai/phobert-large",
    "xlm_roberta_base": "xlm-roberta-base",
    "xlm_roberta_large": "xlm-roberta-large"
}

TOKENIZER_MAP = {
    "phobert_base": "vinai/phobert-base",
    "phobert_large": "vinai/phobert-base",
    "xlm_roberta_base": "xlm-roberta-base",
    "xlm_roberta_large": "xlm-roberta-large",
}


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma

    def forward(self, logits, targets):
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
        lr: float = 1e-5,
        head_lr: float = 1e-4,
        epochs: int = 5,
        weight_decay: float = 0.01,
        save_dir: str = "checkpoints",
        warmup_ratio: float = 0.1,
        class_weight_power: float = 1.0,
        max_class_weight: float = 0.0,
        sentiment_loss_weight: float = 1.0,
        structured_joint_loss_weight: float = 0.0,
        constrained_joint_decoding: bool = False,
        early_stopping_patience: int = 0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.loss_key = loss_key
        self.save_dir = save_dir
        self.class_weight_power = class_weight_power
        self.max_class_weight = max_class_weight
        self.sentiment_loss_weight = sentiment_loss_weight
        self.structured_joint_loss_weight = structured_joint_loss_weight
        self.constrained_joint_decoding = constrained_joint_decoding
        self.early_stopping_patience = early_stopping_patience
        self.vocab = getattr(train_loader.dataset, "vocab", None)

        if self.vocab is None:
            raise ValueError("Train dataset phải cung cấp vocabulary.")

        self.label_aspect_ids, self.label_sentiment_ids = build_label_indices(
            self.vocab,
            self.device,
        )

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        no_decay = ['bias', 'LayerNorm.weight']
        encoder_params = []
        encoder_no_decay_params = []
        head_params = []
        head_no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            is_no_decay = any(nd in name for nd in no_decay)
            is_encoder = name.startswith("encoder.")

            if is_encoder and is_no_decay:
                encoder_no_decay_params.append(param)
            elif is_encoder:
                encoder_params.append(param)
            elif is_no_decay:
                head_no_decay_params.append(param)
            else:
                head_params.append(param)

        optimizer_grouped_parameters = [
            {'params': encoder_params, 'weight_decay': weight_decay, 'lr': lr},
            {'params': encoder_no_decay_params, 'weight_decay': 0.0, 'lr': lr},
            {'params': head_params, 'weight_decay': weight_decay, 'lr': head_lr},
            {'params': head_no_decay_params, 'weight_decay': 0.0, 'lr': head_lr},
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters)

        total_steps = len(train_loader) * epochs
        warmup_steps = int(warmup_ratio * total_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        self.aspect_weights, self.sentiment_weights, self.label_weights = self._build_class_weights()
        self.criterion = nn.CrossEntropyLoss()
        self.aspect_weighted_criterion = nn.CrossEntropyLoss(weight=self.aspect_weights)
        self.sentiment_weighted_criterion = nn.CrossEntropyLoss(weight=self.sentiment_weights)
        self.label_weighted_criterion = nn.CrossEntropyLoss(weight=self.label_weights)
        self.aspect_focal = FocalLoss(weight=self.aspect_weights, gamma=2.0)
        self.sentiment_focal = FocalLoss(weight=self.sentiment_weights, gamma=2.0)
        self.aspect_soft_focal = FocalLoss(weight=None, gamma=1.0)
        self.sentiment_soft_focal = FocalLoss(weight=None, gamma=1.0)
        self.label_to_aspect, self.label_to_sentiment = self._build_label_projection()

    def _make_weights(self, ids, num_classes):
        counts = torch.zeros(num_classes, dtype=torch.float)
        for idx in ids:
            counts[int(idx)] += 1.0

        nonzero = counts > 0
        weights = torch.zeros_like(counts)
        weights[nonzero] = counts[nonzero].pow(-self.class_weight_power)
        weights[nonzero] /= weights[nonzero].mean().clamp(min=1e-6)

        if self.max_class_weight > 0:
            min_weight = 1.0 / self.max_class_weight
            weights[nonzero] = weights[nonzero].clamp(
                min=min_weight,
                max=self.max_class_weight,
            )

        weights[counts == 0] = 0.0
        return weights.to(self.device)

    def _build_class_weights(self):
        dataset = self.train_loader.dataset
        samples = getattr(dataset, "samples", [])
        vocab = getattr(dataset, "vocab", None)

        if vocab is None or not samples:
            return None, None, None

        aspect_ids = [vocab.encode_aspect(sample["aspect"]) for sample in samples]
        sentiment_ids = [vocab.encode_sentiment(sample["sentiment"]) for sample in samples]
        label_ids = [vocab.encode_label(sample["label"]) for sample in samples]

        aspect_weights = self._make_weights(aspect_ids, vocab.num_aspects)
        sentiment_weights = self._make_weights(sentiment_ids, vocab.num_sentiments)
        label_weights = self._make_weights(label_ids, vocab.num_labels)
        return aspect_weights, sentiment_weights, label_weights

    def _build_label_projection(self):
        dataset = self.train_loader.dataset
        vocab = getattr(dataset, "vocab", None)
        if vocab is None:
            return None, None

        label_to_aspect = torch.zeros(vocab.num_labels, vocab.num_aspects, dtype=torch.float)
        label_to_sentiment = torch.zeros(vocab.num_labels, vocab.num_sentiments, dtype=torch.float)

        for label_id, label in vocab.id2label.items():
            aspect, sentiment = vocab.split_label(label)
            label_to_aspect[label_id, vocab.encode_aspect(aspect)] = 1.0
            label_to_sentiment[label_id, vocab.encode_sentiment(sentiment)] = 1.0

        return label_to_aspect.to(self.device), label_to_sentiment.to(self.device)

    def _joint_consistency_loss(self, outputs):
        aspect_probs = torch.softmax(outputs["aspect_logits"], dim=-1)
        sentiment_probs = torch.softmax(outputs["sentiment_logits"], dim=-1)
        joint_probs = torch.softmax(outputs["label_logits"], dim=-1)

        aspect_from_joint = torch.matmul(joint_probs, self.label_to_aspect)
        sentiment_from_joint = torch.matmul(joint_probs, self.label_to_sentiment)

        aspect_consistency = F.mse_loss(aspect_probs, aspect_from_joint)
        sentiment_consistency = F.mse_loss(sentiment_probs, sentiment_from_joint)
        return aspect_consistency + sentiment_consistency

    def compute_loss(self, outputs, batch):
        """Hàm Loss linh hoạt dựa trên loss_key từ config"""
        # --- CẤU TRÚC 1: Multitask CE (Mặc định) ---
        if self.loss_key == "multitask_ce":
            loss_a = self.criterion(outputs["aspect_logits"], batch["aspect_id"].to(self.device))
            loss_s = self.criterion(outputs["sentiment_logits"], batch["sentiment_id"].to(self.device))
            return loss_a + loss_s

        # --- CẤU TRÚC 2: Weighted Multitask CE ---
        elif self.loss_key == "multitask_weighted_ce":
            loss_a = self.aspect_weighted_criterion(outputs["aspect_logits"], batch["aspect_id"].to(self.device))
            loss_s = self.sentiment_weighted_criterion(outputs["sentiment_logits"], batch["sentiment_id"].to(self.device))
            loss = loss_a + self.sentiment_loss_weight * loss_s

            if self.structured_joint_loss_weight > 0:
                joint_logits = structured_joint_logits(
                    outputs["aspect_logits"],
                    outputs["sentiment_logits"],
                    self.label_aspect_ids,
                    self.label_sentiment_ids,
                )
                loss_j = self.label_weighted_criterion(
                    joint_logits,
                    batch["label_id"].to(self.device),
                )
                loss = loss + self.structured_joint_loss_weight * loss_j

            return loss

        # --- CẤU TRÚC 3: Focal Loss cho dữ liệu lệch lớp ---
        elif self.loss_key == "focal_multitask":
            loss_a = self.aspect_focal(outputs["aspect_logits"], batch["aspect_id"].to(self.device))
            loss_s = self.sentiment_focal(outputs["sentiment_logits"], batch["sentiment_id"].to(self.device))
            return loss_a + loss_s

        # --- CẤU TRÚC 3b: Focal nhẹ, tránh over-focus vào rare/noisy labels ---
        elif self.loss_key == "focal_multitask_soft":
            loss_a = self.aspect_soft_focal(outputs["aspect_logits"], batch["aspect_id"].to(self.device))
            loss_s = self.sentiment_soft_focal(outputs["sentiment_logits"], batch["sentiment_id"].to(self.device))
            return loss_a + loss_s
            
        # --- CẤU TRÚC 4: Joint Label ---
        elif self.loss_key == "joint_ce":
            return self.label_weighted_criterion(outputs["label_logits"], batch["label_id"].to(self.device))

        # --- CẤU TRÚC 5: Multitask + auxiliary full-label head ---
        elif self.loss_key == "joint_aux_loss":
            loss_a = self.aspect_weighted_criterion(outputs["aspect_logits"], batch["aspect_id"].to(self.device))
            loss_s = self.sentiment_weighted_criterion(outputs["sentiment_logits"], batch["sentiment_id"].to(self.device))
            loss_j = self.label_weighted_criterion(outputs["label_logits"], batch["label_id"].to(self.device))
            return loss_a + loss_s + 0.5 * loss_j

        # --- CẤU TRÚC 5b: Joint auxiliary CE nhẹ, ổn định hơn weighted joint aux ---
        elif self.loss_key == "joint_aux_ce":
            loss_a = self.criterion(outputs["aspect_logits"], batch["aspect_id"].to(self.device))
            loss_s = self.criterion(outputs["sentiment_logits"], batch["sentiment_id"].to(self.device))
            loss_j = self.criterion(outputs["label_logits"], batch["label_id"].to(self.device))
            return loss_a + loss_s + 0.3 * loss_j

        # --- CẤU TRÚC 5c: Tối ưu joint F1 bằng consistency giữa joint/aspect/sentiment ---
        elif self.loss_key == "joint_consistency_loss":
            loss_a = self.criterion(outputs["aspect_logits"], batch["aspect_id"].to(self.device))
            loss_s = self.criterion(outputs["sentiment_logits"], batch["sentiment_id"].to(self.device))
            loss_j = self.criterion(outputs["label_logits"], batch["label_id"].to(self.device))
            loss_c = self._joint_consistency_loss(outputs)
            return loss_a + loss_s + 0.4 * loss_j + 0.2 * loss_c

        raise ValueError(f"Loss key chưa được hỗ trợ: {self.loss_key}")

    def _forward_pass(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        aspect_ids = batch.get("aspect_id", None)
        if aspect_ids is not None: aspect_ids = aspect_ids.to(self.device)

        target_mask = batch.get("target_mask", None)
        if target_mask is not None: target_mask = target_mask.to(self.device)

        sentence_mask = batch.get("sentence_mask", None)
        if sentence_mask is not None: sentence_mask = sentence_mask.to(self.device)

        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None: token_type_ids = token_type_ids.to(self.device)

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            aspect_ids=aspect_ids,
            target_mask=target_mask,
            sentence_mask=sentence_mask,
        )

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]", leave=False)
        for batch in progress_bar:
            self.optimizer.zero_grad()
            outputs = self._forward_pass(batch)
            loss = self.compute_loss(outputs, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        all_aspect_preds, all_aspect_targets = [], []
        all_sentiment_preds, all_sentiment_targets = [], []
        all_joint_preds, all_joint_targets = [], []
        val_loss = 0

        progress_bar = tqdm(self.val_loader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for batch in progress_bar:
                outputs = self._forward_pass(batch)
                loss = self.compute_loss(outputs, batch)
                val_loss += loss.item()

                if "aspect_logits" in outputs and "sentiment_logits" in outputs:
                    if self.constrained_joint_decoding:
                        aspect_preds, sentiment_preds, label_preds = decode_structured_joint(
                            outputs["aspect_logits"],
                            outputs["sentiment_logits"],
                            self.label_aspect_ids,
                            self.label_sentiment_ids,
                        )
                        batch_joint_preds = [
                            self.vocab.id2label[int(label_id)]
                            for label_id in label_preds.cpu().tolist()
                        ]
                    else:
                        aspect_preds = torch.argmax(outputs["aspect_logits"], dim=-1)
                        sentiment_preds = torch.argmax(outputs["sentiment_logits"], dim=-1)
                        batch_joint_preds = [
                            f"{self.vocab.id2aspect[int(a)]}#{self.vocab.id2sentiment[int(s)]}"
                            for a, s in zip(
                                aspect_preds.cpu().tolist(),
                                sentiment_preds.cpu().tolist(),
                            )
                        ]

                    all_aspect_preds.extend(aspect_preds.cpu().tolist())
                    all_sentiment_preds.extend(sentiment_preds.cpu().tolist())
                    all_aspect_targets.extend(batch["aspect_id"].numpy())
                    all_sentiment_targets.extend(batch["sentiment_id"].numpy())
                    all_joint_preds.extend(batch_joint_preds)
                    all_joint_targets.extend([
                        self.vocab.id2label[int(label_id)]
                        for label_id in batch["label_id"].tolist()
                    ])

                elif "label_logits" in outputs:
                    all_joint_preds.extend(torch.argmax(outputs["label_logits"], dim=-1).cpu().numpy())
                    all_joint_targets.extend(batch["label_id"].numpy())

        # An toàn nếu mô hình không trả về aspect/sentiment (VD: joint_label)
        if len(all_joint_targets) == 0:
            return {"val_loss": val_loss / len(self.val_loader), "macro_f1": 0.0, "joint_macro_f1": 0.0}

        metrics = {
            "val_loss": val_loss / len(self.val_loader),
            "joint_macro_f1": f1_score(all_joint_targets, all_joint_preds, average='macro', zero_division=0),
        }
        metrics["macro_f1"] = metrics["joint_macro_f1"]

        if len(all_aspect_targets) > 0:
            metrics["aspect_macro_f1"] = f1_score(all_aspect_targets, all_aspect_preds, average='macro', zero_division=0)
            metrics["sentiment_macro_f1"] = f1_score(all_sentiment_targets, all_sentiment_preds, average='macro', zero_division=0)

        return metrics

    def train(self):
        best_f1 = -1.0
        epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            metrics = self.evaluate()
            
            print(f"   Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {metrics['val_loss']:.4f} | F1: {metrics['macro_f1']:.4f}")

            if metrics['macro_f1'] > best_f1:
                best_f1 = metrics['macro_f1']
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model.pt"))
            else:
                epochs_without_improvement += 1

            if (
                self.early_stopping_patience > 0
                and epochs_without_improvement >= self.early_stopping_patience
            ):
                print(f"   Early stopping sau {epoch} epochs.")
                break
        
        print(f"Hoàn thành! Best F1: {best_f1:.4f}. Đã lưu tại: {self.save_dir}/best_model.pt\n")


# ==========================================
# TRÌNH QUẢN LÝ THÍ NGHIỆM TỪ YAML
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train targeted ABSA experiments")
    parser.add_argument("--domain", choices=["hotel", "restaurant", "combined"], default="hotel")
    parser.add_argument("--run-name", default=None, help="Chỉ train run có tên này.")
    parser.add_argument("--config", default="config/experiments.yaml")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    if args.domain == "combined":
        train_data = "Data/train.jsonl"
        val_data = "Data/dev.jsonl"
    else:
        train_data = f"Data/{args.domain}_train.jsonl"
        val_data = f"Data/{args.domain}_dev.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    print(f"Thiết bị: {device}")
    print(f"Domain: {args.domain} | Train: {train_data} | Dev: {val_data}")

    # 1. ĐỌC FILE YAML
    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy file {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    experiment_groups = config.get("experiment_groups", {})
    
    # Đếm tổng số lượng run
    total_runs = sum(
        1
        for group in experiment_groups.values()
        for run in group.get("runs", [])
        if args.run_name is None or run.get("name") == args.run_name
    )
    if total_runs == 0:
        raise ValueError(f"Không tìm thấy run '{args.run_name}' trong {config_path}")
    print(f"Tìm thấy {len(experiment_groups)} nhóm thí nghiệm. Tổng cộng {total_runs} runs.\n")

    # 2. VÒNG LẶP QUA TỪNG NHÓM VÀ TỪNG RUN
    for group_name, group_data in experiment_groups.items():
        print("="*70)
        print(f"BẮT ĐẦU NHÓM THÍ NGHIỆM: {group_name.upper()}")
        print(f"{group_data.get('description', '')}")
        print("="*70)

        runs = group_data.get("runs", [])
        for run in runs:
            run_name = run["name"]
            if args.run_name is not None and run_name != args.run_name:
                continue

            enc_key = run.get("encoder_key", "phobert_base")
            attn_key = run.get("attention_key", "cls_pooling")
            label_key = run.get("label_structure_key", "multitask_aspect_sentiment")
            loss_key = run.get("loss_key", "multitask_ce")
            
            # Ánh xạ encoder
            encoder_name = ENCODER_MAP.get(enc_key, "vinai/phobert-base")
            tokenizer_name = TOKENIZER_MAP.get(enc_key, encoder_name)

            print(f"\n RUN: {run_name}")
            print(f"   - Encoder: {encoder_name}")
            print(f"   - Label Struct: {label_key}")
            print(f"   - Attention: {attn_key}")
            print(f"   - Loss: {loss_key}")

            # Thư mục lưu checkpoint: checkpoints/encoder_comparison/enc_phobert_base/
            save_dir = os.path.join("checkpoints", args.domain, group_name, run_name)

            model = None
            trainer = None
            try:
                vocab = Vocabulary(
                    data_path=[train_data, val_data],
                    model_name=tokenizer_name,
                )
                batch_size = int(run.get("batch_size", 16))
                max_length = int(run.get("max_length", 128))
                train_loader = build_dataloader(
                    train_data,
                    vocab,
                    batch_size,
                    max_length,
                    shuffle=True,
                    pin_memory=torch.cuda.is_available(),
                )
                val_loader = build_dataloader(
                    val_data,
                    vocab,
                    batch_size,
                    max_length,
                    shuffle=False,
                    pin_memory=torch.cuda.is_available(),
                )

                # Khởi tạo mô hình
                model = TargetedABSAModel(
                    encoder_name=encoder_name,
                    num_aspects=vocab.num_aspects,
                    num_sentiments=vocab.num_sentiments,
                    num_labels=vocab.num_labels, # Dùng cho joint_label
                    label_structure_key=label_key,
                    attention_key=attn_key,
                    dropout=float(run.get("dropout", 0.1)),
                )

                # Khởi tạo Trainer
                trainer = ABSATrainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    loss_key=loss_key,
                    lr=float(run.get("encoder_lr", 1e-5)),
                    head_lr=float(run.get("head_lr", 1e-4)),
                    epochs=int(run.get("epochs", 15)),
                    weight_decay=float(run.get("weight_decay", 0.01)),
                    save_dir=save_dir,
                    warmup_ratio=float(run.get("warmup_ratio", 0.1)),
                    class_weight_power=float(run.get("class_weight_power", 1.0)),
                    max_class_weight=float(run.get("max_class_weight", 0.0)),
                    sentiment_loss_weight=float(run.get("sentiment_loss_weight", 1.0)),
                    structured_joint_loss_weight=float(run.get("structured_joint_loss_weight", 0.0)),
                    constrained_joint_decoding=bool(run.get("constrained_joint_decoding", False)),
                    early_stopping_patience=int(run.get("early_stopping_patience", 0)),
                )

                # Chạy huấn luyện
                trainer.train()

            except Exception as e:
                print(f"LỖI TẠI RUN '{run_name}': {str(e)}")
                print("Bỏ qua và chạy thí nghiệm tiếp theo...")

            # Giải phóng RAM/VRAM
            if model is not None:
                del model
            if trainer is not None:
                del trainer
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("\nĐÃ HOÀN THÀNH CÁC THÍ NGHIỆM ĐÃ CHỌN!")

if __name__ == "__main__":
    main()
