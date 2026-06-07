import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from vocabulary import Vocabulary
from dataloader import ABSATargetedDataset, build_dataloader
from model.targeted_absa import TargetedABSAModel

transformers.logging.set_verbosity_error()

class ABSATrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 2e-5,
        epochs: int = 5,
        weight_decay: float = 0.01,
        aspect_loss_weight: float = 1.0,
        sentiment_loss_weight: float = 1.0,
        save_dir: str = "checkpoints"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.aspect_loss_weight = aspect_loss_weight
        self.sentiment_loss_weight = sentiment_loss_weight
        self.save_dir = save_dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Khởi tạo Optimizer chuẩn cho Transformer
        # Loại bỏ weight decay cho các tham số bias và LayerNorm
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

        # Cài đặt Learning Rate Scheduler (Linear Warmup)
        total_steps = len(train_loader) * epochs
        warmup_steps = int(0.1 * total_steps) # 10% thời gian đầu để warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Hàm Loss
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, outputs, batch):
        """Tính toán tổng Loss cho kiến trúc Multi-task"""
        aspect_logits = outputs["aspect_logits"]
        sentiment_logits = outputs["sentiment_logits"]
        
        aspect_targets = batch["aspect_id"].to(self.device)
        sentiment_targets = batch["sentiment_id"].to(self.device)

        loss_aspect = self.criterion(aspect_logits, aspect_targets)
        loss_sentiment = self.criterion(sentiment_logits, sentiment_targets)

        # Trọng số loss: Có thể sentiment dễ đoán hơn aspect nên cần cân bằng
        total_loss = (self.aspect_loss_weight * loss_aspect) + (self.sentiment_loss_weight * loss_sentiment)
        return total_loss, loss_aspect, loss_sentiment

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")
        for batch in progress_bar:
            self.optimizer.zero_grad()

            # Đẩy dữ liệu lên GPU/CPU
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Tính Loss
            loss, loss_a, loss_s = self.compute_loss(outputs, batch)
            
            # Backward pass & Tối ưu hóa
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # Tránh nổ gradient
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item(), 'lr': self.scheduler.get_last_lr()[0]})

        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        
        all_aspect_preds, all_aspect_targets = [], []
        all_sentiment_preds, all_sentiment_targets = [], []
        val_loss = 0

        progress_bar = tqdm(self.val_loader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                loss, _, _ = self.compute_loss(outputs, batch)
                val_loss += loss.item()

                # Lấy nhãn dự đoán bằng argmax
                aspect_preds = torch.argmax(outputs["aspect_logits"], dim=-1).cpu().numpy()
                sentiment_preds = torch.argmax(outputs["sentiment_logits"], dim=-1).cpu().numpy()

                all_aspect_preds.extend(aspect_preds)
                all_sentiment_preds.extend(sentiment_preds)
                all_aspect_targets.extend(batch["aspect_id"].numpy())
                all_sentiment_targets.extend(batch["sentiment_id"].numpy())

        # Tính toán Metrics
        metrics = {
            "val_loss": val_loss / len(self.val_loader),
            "aspect_acc": accuracy_score(all_aspect_targets, all_aspect_preds),
            "aspect_f1": f1_score(all_aspect_targets, all_aspect_preds, average='macro'),
            "sentiment_acc": accuracy_score(all_sentiment_targets, all_sentiment_preds),
            "sentiment_f1": f1_score(all_sentiment_targets, all_sentiment_preds, average='macro')
        }
        return metrics

    def train(self):
        best_f1 = 0.0 # Theo dõi dựa trên F1 trung bình của cả Aspect và Sentiment
        
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            metrics = self.evaluate()
            
            avg_macro_f1 = (metrics['aspect_f1'] + metrics['sentiment_f1']) / 2

            print(f"\n--- Result Epoch {epoch} ---")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {metrics['val_loss']:.4f}")
            print(f"Aspect    -> Acc: {metrics['aspect_acc']:.4f} | Macro-F1: {metrics['aspect_f1']:.4f}")
            print(f"Sentiment -> Acc: {metrics['sentiment_acc']:.4f} | Macro-F1: {metrics['sentiment_f1']:.4f}")
            print(f"Avg F1    -> {avg_macro_f1:.4f}\n")

            # Lưu mô hình tốt nhất
            if avg_macro_f1 > best_f1:
                best_f1 = avg_macro_f1
                save_path = os.path.join(self.save_dir, "best_model.pt")
                torch.save(self.model.state_dict(), save_path)
                print(f"Đã lưu model tốt nhất (F1: {best_f1:.4f}) tại {save_path}\n")

# ==========================================
# HÀM MAIN CHẠY CHƯƠNG TRÌNH
# ==========================================
def main():
    # 1. Khai báo các đường dẫn và thiết lập
    MODEL_NAME = "vinai/phobert-base"
    TRAIN_DATA = "Data/train.jsonl"
    VAL_DATA = "Data/dev.jsonl"
    BATCH_SIZE = 16
    MAX_LENGTH = 128
    EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🚀 Đang chạy trên device: {DEVICE}")

    # 2. Khởi tạo Vocabulary (Phải dùng cả train và val để quét hết nhãn)
    print("Trích xuất từ điển...")
    vocab = Vocabulary(data_path=[TRAIN_DATA, VAL_DATA], model_name=MODEL_NAME)
    print(f"-> Tìm thấy {vocab.num_aspects} aspects và {vocab.num_sentiments} sentiments.")

    # 3. Tạo DataLoader
    print("Chuẩn bị DataLoader...")
    train_loader = build_dataloader(
        data_path=TRAIN_DATA, vocab=vocab, batch_size=BATCH_SIZE, 
        max_length=MAX_LENGTH, shuffle=True
    )
    val_loader = build_dataloader(
        data_path=VAL_DATA, vocab=vocab, batch_size=BATCH_SIZE, 
        max_length=MAX_LENGTH, shuffle=False
    )

    # 4. Khởi tạo Model
    print("Khởi tạo mô hình Targeted ABSA...")
    model = TargetedABSAModel(
        encoder_name=MODEL_NAME,
        num_aspects=vocab.num_aspects,
        num_sentiments=vocab.num_sentiments,
        label_structure_key="multitask_aspect_sentiment",
        attention_key="target_conditioned_attention", # Chọn cơ chế attention tốt nhất của bạn
        dropout=0.1
    )

    # 5. Khởi tạo Trainer và bắt đầu huấn luyện
    trainer = ABSATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        lr=2e-5,
        epochs=EPOCHS,
        aspect_loss_weight=1.0,     # Có thể chỉnh (vd: 0.7) nếu aspect khó hội tụ
        sentiment_loss_weight=1.0   # Có thể chỉnh (vd: 0.3)
    )

    trainer.train()

if __name__ == "__main__":
    main()
