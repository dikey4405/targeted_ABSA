import json
import torch
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union
from transformers import AutoTokenizer

class Vocabulary:
    def __init__(self, data_path: Union[str, Sequence[str]], model_name: str):
        super().__init__()
        self.data_path = data_path
        # Khởi tạo pretrained tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        label_set = set()
        aspect_set = set()
        sentiment_set = set()

        # Duyệt qua dữ liệu để thu thập tập hợp các nhãn ABSA.
        for item in self._read_data():
            sample_labels = self.get_labels_from_sample(item)
            for full_label in sample_labels:
                label_set.add(full_label)
                
                aspect, sentiment = self.split_label(full_label)
                aspect_set.add(aspect)
                sentiment_set.add(sentiment)

        # Sắp xếp cố định danh sách nhãn
        self.aspects = sorted(list(aspect_set))
        self.sentiments = sorted(list(sentiment_set))
        self.labels = sorted(list(label_set))

        # Tạo các từ điển ánh xạ hai chiều (String <-> ID)
        self.aspect2id = {aspect: idx for idx, aspect in enumerate(self.aspects)}
        self.id2aspect = {idx: aspect for aspect, idx in self.aspect2id.items()}

        self.sentiment2id = {sentiment: idx for idx, sentiment in enumerate(self.sentiments)}
        self.id2sentiment = {idx: sentiment for sentiment, idx in self.sentiment2id.items()}

        self.label2id = {label: idx for idx, label in enumerate(self.labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def _read_data(self) -> Iterable[Dict[str, Any]]:
        """Đọc dữ liệu từ file JSON Lines; vẫn hỗ trợ JSON list/dict nếu cần."""
        paths = [self.data_path] if isinstance(self.data_path, (str, Path)) else self.data_path

        for path in paths:
            with open(path, "r", encoding="utf-8-sig") as f:
                text = f.read().strip()

            if not text:
                continue

            try:
                raw = json.loads(text)
                if isinstance(raw, list):
                    for item in raw:
                        if isinstance(item, dict):
                            yield item
                elif isinstance(raw, dict):
                    for item in raw.values():
                        if isinstance(item, dict):
                            yield item
                continue
            except json.JSONDecodeError:
                pass

            with open(path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            yield item

    def normalize_label(self, label: Any) -> str:
        return str(label).strip().replace(" ", "").upper()

    def split_label(self, label: Any) -> Tuple[str, str]:
        """
        Tách nhãn đầy đủ thành aspect và sentiment.
        Ví dụ: ROOMS#CLEANLINESS#POSITIVE -> (ROOMS#CLEANLINESS, POSITIVE)
        """
        label = self.normalize_label(label)
        if "#" not in label:
            raise ValueError(f"Invalid ABSA label: {label}")
        aspect, sentiment = label.rsplit("#", 1)
        return aspect, sentiment

    def get_labels_from_sample(self, sample: Dict[str, Any]) -> List[str]:
        """
        Trích xuất danh sách các chuỗi nhãn đầy đủ từ trường 'label' của một sample.
        Ví dụ đầu ra: ['ROOMS#CLEANLINESS#POSITIVE', 'SERVICE#GENERAL#POSITIVE']
        """
        extracted_labels = []
        for annotation in sample.get("label", []):
            if isinstance(annotation, (list, tuple)) and len(annotation) >= 3:
                extracted_labels.append(self.normalize_label(annotation[2]))
        return extracted_labels

    def get_aspects_from_sample(self, sample: Dict[str, Any]) -> List[str]:
        return [self.split_label(label)[0] for label in self.get_labels_from_sample(sample)]

    def get_sentiments_from_sample(self, sample: Dict[str, Any]) -> List[str]:
        return [self.split_label(label)[1] for label in self.get_labels_from_sample(sample)]

    def get_targets_from_sample(self, sample: Dict[str, Any]) -> List[str]:
        text = sample.get("data", "")
        if not isinstance(text, str):
            return []

        targets = []
        for annotation in sample.get("label", []):
            if (
                isinstance(annotation, (list, tuple))
                and len(annotation) >= 3
                and isinstance(annotation[0], int)
                and isinstance(annotation[1], int)
            ):
                targets.append(text[annotation[0]:annotation[1]])
        return targets

    def get_annotations_from_sample(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Trả về annotation đã tách rõ target, aspect, sentiment để dùng cho ABSA targeted.
        """
        text = sample.get("data", "")
        annotations = []

        for annotation in sample.get("label", []):
            if (
                not isinstance(annotation, (list, tuple))
                or len(annotation) < 3
                or not isinstance(annotation[0], int)
                or not isinstance(annotation[1], int)
            ):
                continue

            start, end = annotation[0], annotation[1]
            label = self.normalize_label(annotation[2])
            aspect, sentiment = self.split_label(label)
            target = text[start:end] if isinstance(text, str) else ""

            annotations.append({
                "start": start,
                "end": end,
                "target": target,
                "label": label,
                "aspect": aspect,
                "sentiment": sentiment,
            })

        return annotations

    def encode_label(self, label: Any) -> int:
        label = self.normalize_label(label)
        if label not in self.label2id:
            raise KeyError(f"Unknown label: {label}")
        return self.label2id[label]

    def encode_aspect(self, aspect: Any) -> int:
        aspect = self.normalize_label(aspect)
        if aspect not in self.aspect2id:
            raise KeyError(f"Unknown aspect: {aspect}")
        return self.aspect2id[aspect]

    def encode_sentiment(self, sentiment: Any) -> int:
        sentiment = self.normalize_label(sentiment)
        if sentiment not in self.sentiment2id:
            raise KeyError(f"Unknown sentiment: {sentiment}")
        return self.sentiment2id[sentiment]

    def encode_multilabel(self, labels: List[str]) -> torch.Tensor:
        """
        Chuyển đổi danh sách các chuỗi nhãn thành một Tensor nhị phân (Multi-hot vector).
        Kích thước vector bằng tổng số lượng nhãn độc nhất thu được từ toàn bộ dataset.
        """
        multi_hot = torch.zeros(len(self.labels), dtype=torch.float32)
        
        for label in labels:
            label = self.normalize_label(label)
            if label in self.label2id:
                multi_hot[self.label2id[label]] = 1.0
                
        return multi_hot

    def tokenize_text(self, text: str, max_length: int = 256) -> Dict[str, torch.Tensor]:
        """
        Mã hóa chuỗi văn bản đầu vào bằng PhoBERT Tokenizer.
        Trả về một dictionary chứa 'input_ids' và 'attention_mask' dạng PyTorch Tensor.
        """
        inputs = self.tokenizer(
            text,
            padding="max_length",       
            truncation="only_first",            
            max_length=max_length,
            return_tensors="pt"         
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}

    def tokenize_text_pair(self, text: str, target: str, max_length: int = 256) -> Dict[str, torch.Tensor]:
        """
        Mã hóa cặp (sentence, target) cho mô hình targeted ABSA.
        """
        inputs = self.tokenizer(
            text,
            target,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        encoded = {k: v.squeeze(0) for k, v in inputs.items()}
        special_tokens_mask = encoded.pop("special_tokens_mask")
        encoded["target_mask"] = self._build_target_mask(encoded["input_ids"], target)
        encoded["sentence_mask"] = (
            encoded["attention_mask"].bool()
            & ~encoded["target_mask"].bool()
            & ~special_tokens_mask.bool()
        ).long()
        return encoded

    def _build_target_mask(self, input_ids: torch.Tensor, target: str) -> torch.Tensor:
        """
        Đánh dấu các token của target trong input pair.
        Cách này không phụ thuộc token_type_ids, nên dùng được cho PhoBERT/XLM-R.
        """
        target_ids = self.tokenizer(
            target,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"].squeeze(0)

        mask = torch.zeros_like(input_ids, dtype=torch.long)
        if target_ids.numel() == 0 or target_ids.numel() > input_ids.numel():
            return mask

        input_list = input_ids.tolist()
        target_list = target_ids.tolist()
        target_len = len(target_list)

        # Với text pair, target thường nằm ở nửa sau chuỗi; tìm từ cuối để tránh
        # trùng với cùng cụm từ xuất hiện trong sentence.
        for start in range(len(input_list) - target_len, -1, -1):
            if input_list[start:start + target_len] == target_list:
                mask[start:start + target_len] = 1
                break

        return mask

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    @property
    def num_aspects(self) -> int:
        return len(self.aspects)

    @property
    def num_sentiments(self) -> int:
        return len(self.sentiments)

    def __len__(self) -> int:
        return len(self.labels)
