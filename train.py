import os
import yaml
import torch
import torch.nn as nn
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

        # Base Criterion (có thể mở rộng dựa trên loss_key)
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, outputs, batch):
        """Hàm Loss linh hoạt dựa trên loss_key từ config"""
        # --- CẤU TRÚC 1: Multitask CE (Mặc định) ---
        if self.loss_key in ["multitask_ce", "multitask_weighted_ce"]:
            loss_a = self.criterion(outputs["aspect_logits"], batch["aspect_id"].to(self.device))
            loss_s = self.criterion(outputs["sentiment_logits"], batch["sentiment_id"].to(self.device))
            
            # Nếu là weighted, ta có thể đổi trọng số ở đây
            weight_a = 1.0
            weight_s = 1.0 if self.loss_key == "multitask_ce" else 0.5 
            
            total_loss = (weight_a * loss_a) + (weight_s * loss_s)
            return total_loss
            
        # --- CẤU TRÚC 2: Joint Label ---
        elif self.loss_key == "joint_ce":
            # Giả định outputs["label_logits"] được thiết lập trong model
            return self.criterion(outputs["label_logits"], batch["label_id"].to(self.device))

        # --- TODO: Thêm các cấu trúc Grid/Focal Loss tại đây sau ---
        else:
            # Fallback về mặc định nếu chưa code xong loss đó
            loss_a = self.criterion(outputs.get("aspect_logits"), batch["aspect_id"].to(self.device))
            loss_s = self.criterion(outputs.get("sentiment_logits"), batch["sentiment_id"].to(self.device))
            return loss_a + loss_s

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

        # An toàn nếu mô hình không trả về aspect/sentiment (VD: joint_label)
        if len(all_aspect_targets) == 0:
            return {"val_loss": val_loss / len(self.val_loader), "macro_f1": 0.0}

        metrics = {
            "val_loss": val_loss / len(self.val_loader),
            "macro_f1": (f1_score(all_aspect_targets, all_aspect_preds, average='macro') + 
                         f1_score(all_sentiment_targets, all_sentiment_preds, average='macro')) / 2
        }
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
    EPOCHS = 10
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

    print("\n🎉 ĐÃ HOÀN THÀNH TOÀN BỘ CÁC THÍ NGHIỆM TRONG CONFIG!")

if __name__ == "__main__":
    main()
