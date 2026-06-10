import os
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
        lr: float = 2e-5,
        epochs: int = 5,
        weight_decay: float = 0.01,
        save_dir: str = "checkpoints"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.loss_key = loss_key
        self.save_dir = save_dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

        total_steps = len(train_loader) * epochs
        warmup_steps = int(0.1 * total_steps)
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

    def _make_weights(self, ids, num_classes):
        counts = torch.zeros(num_classes, dtype=torch.float)
        for idx in ids:
            counts[int(idx)] += 1.0

        weights = counts.sum() / counts.clamp(min=1.0)
        weights = weights / weights.mean().clamp(min=1e-6)
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
            return loss_a + loss_s

        # --- CẤU TRÚC 3: Focal Loss cho dữ liệu lệch lớp ---
        elif self.loss_key == "focal_multitask":
            loss_a = self.aspect_focal(outputs["aspect_logits"], batch["aspect_id"].to(self.device))
            loss_s = self.sentiment_focal(outputs["sentiment_logits"], batch["sentiment_id"].to(self.device))
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

        raise ValueError(f"Loss key chưa được hỗ trợ: {self.loss_key}")

    def _forward_pass(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        aspect_ids = batch.get("aspect_id", None)
        if aspect_ids is not None: aspect_ids = aspect_ids.to(self.device)

        target_mask = batch.get("target_mask", None)
        if target_mask is not None: target_mask = target_mask.to(self.device)

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            aspect_ids=aspect_ids,
            target_mask=target_mask
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
                    all_aspect_preds.extend(torch.argmax(outputs["aspect_logits"], dim=-1).cpu().numpy())
                    all_sentiment_preds.extend(torch.argmax(outputs["sentiment_logits"], dim=-1).cpu().numpy())
                    all_aspect_targets.extend(batch["aspect_id"].numpy())
                    all_sentiment_targets.extend(batch["sentiment_id"].numpy())
                    all_joint_preds.extend([
                        f"{a}#{s}" for a, s in zip(all_aspect_preds[-len(batch["aspect_id"]):], all_sentiment_preds[-len(batch["sentiment_id"]):])
                    ])
                    all_joint_targets.extend([
                        f"{a}#{s}" for a, s in zip(batch["aspect_id"].numpy(), batch["sentiment_id"].numpy())
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
        best_f1 = 0.0 
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            metrics = self.evaluate()
            
            print(f"   Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {metrics['val_loss']:.4f} | F1: {metrics['macro_f1']:.4f}")

            if metrics['macro_f1'] > best_f1:
                best_f1 = metrics['macro_f1']
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model.pt"))
        
        print(f"Hoàn thành! Best F1: {best_f1:.4f}. Đã lưu tại: {self.save_dir}/best_model.pt\n")


# ==========================================
# TRÌNH QUẢN LÝ THÍ NGHIỆM TỪ YAML
# ==========================================
def main():
    # Cấu hình cứng
    TRAIN_DATA = "Data/train.jsonl"
    VAL_DATA = "Data/dev.jsonl"
    BATCH_SIZE = 16
    MAX_LENGTH = 128
    EPOCHS = 15
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Thiết bị: {DEVICE}")

    # Khởi tạo Vocab một lần (Dùng phobert-base làm chuẩn tokenizer, vì XLM-R và PhoBERT đều chung format padding/mask)
    vocab = Vocabulary(data_path=[TRAIN_DATA, VAL_DATA], model_name="vinai/phobert-base")
    
    train_loader = build_dataloader(TRAIN_DATA, vocab, BATCH_SIZE, MAX_LENGTH, shuffle=True)
    val_loader = build_dataloader(VAL_DATA, vocab, BATCH_SIZE, MAX_LENGTH, shuffle=False)

    # 1. ĐỌC FILE YAML
    config_path = "config/experiments.yaml" # Thay bằng đường dẫn file yaml của bạn
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy file {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    experiment_groups = config.get("experiment_groups", {})
    
    # Đếm tổng số lượng run
    total_runs = sum(len(group.get("runs", [])) for group in experiment_groups.values())
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
            enc_key = run.get("encoder_key", "phobert_base")
            attn_key = run.get("attention_key", "cls_pooling")
            label_key = run.get("label_structure_key", "multitask_aspect_sentiment")
            loss_key = run.get("loss_key", "multitask_ce")
            
            # Ánh xạ encoder
            encoder_name = ENCODER_MAP.get(enc_key, "vinai/phobert-base")

            print(f"\n RUN: {run_name}")
            print(f"   - Encoder: {encoder_name}")
            print(f"   - Label Struct: {label_key}")
            print(f"   - Attention: {attn_key}")
            print(f"   - Loss: {loss_key}")

            # Thư mục lưu checkpoint: checkpoints/encoder_comparison/enc_phobert_base/
            save_dir = os.path.join("checkpoints", group_name, run_name)

            try:
                # Khởi tạo mô hình
                model = TargetedABSAModel(
                    encoder_name=encoder_name,
                    num_aspects=vocab.num_aspects,
                    num_sentiments=vocab.num_sentiments,
                    num_labels=vocab.num_labels, # Dùng cho joint_label
                    label_structure_key=label_key,
                    attention_key=attn_key,
                    dropout=0.1
                )

                # Khởi tạo Trainer
                trainer = ABSATrainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=DEVICE,
                    loss_key=loss_key,
                    epochs=EPOCHS,
                    save_dir=save_dir
                )

                # Chạy huấn luyện
                trainer.train()

            except Exception as e:
                print(f"LỖI TẠI RUN '{run_name}': {str(e)}")
                print("Bỏ qua và chạy thí nghiệm tiếp theo...")

            # Giải phóng RAM/VRAM
            del model
            if 'trainer' in locals(): del trainer
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("\n ĐÃ HOÀN THÀNH TOÀN BỘ CÁC THÍ NGHIỆM TRONG CONFIG!")

if __name__ == "__main__":
    main()
