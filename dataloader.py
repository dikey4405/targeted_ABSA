import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader, Dataset


class ABSATargetedDataset(Dataset):
    """
    Dataset cho targeted ABSA.

    Mỗi annotation [start, end, label] trong một review được tách thành một sample:
    (review text, target text) -> full label / aspect / sentiment.
    """

    def __init__(
        self,
        data_path: Union[str, Path, Sequence[Union[str, Path]]],
        vocab,
        max_length: int = 256,
        return_text: bool = False,
    ):
        super().__init__()
        self.vocab = vocab
        self.max_length = max_length
        self.return_text = return_text
        self.samples = self._build_samples(data_path)

    def _read_jsonl(self, data_path: Union[str, Path, Sequence[Union[str, Path]]]) -> Iterable[Dict[str, Any]]:
        paths = [data_path] if isinstance(data_path, (str, Path)) else data_path

        for path in paths:
            with open(path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if isinstance(item, dict):
                        yield item

    def _build_samples(self, data_path: Union[str, Path, Sequence[Union[str, Path]]]) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []

        for item in self._read_jsonl(data_path):
            text = item.get("data", "")
            if not isinstance(text, str):
                continue

            for ann in self.vocab.get_annotations_from_sample(item):
                if not ann["target"]:
                    continue

                samples.append({
                    "id": item.get("id"),
                    "text": text,
                    "start": ann["start"],
                    "end": ann["end"],
                    "target": ann["target"],
                    "label": ann["label"],
                    "aspect": ann["aspect"],
                    "sentiment": ann["sentiment"],
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        encoded = self.vocab.tokenize_text_pair(
            sample["text"],
            sample["target"],
            max_length=self.max_length,
        )

        output: Dict[str, Any] = {
            **encoded,
            "label_id": torch.tensor(self.vocab.encode_label(sample["label"]), dtype=torch.long),
            "aspect_id": torch.tensor(self.vocab.encode_aspect(sample["aspect"]), dtype=torch.long),
            "sentiment_id": torch.tensor(self.vocab.encode_sentiment(sample["sentiment"]), dtype=torch.long),
            "start": torch.tensor(sample["start"], dtype=torch.long),
            "end": torch.tensor(sample["end"], dtype=torch.long),
        }

        if self.return_text:
            output.update({
                "id": sample["id"],
                "text": sample["text"],
                "target": sample["target"],
                "label": sample["label"],
                "aspect": sample["aspect"],
                "sentiment": sample["sentiment"],
            })

        return output


def absa_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate batch gồm tensor và metadata string.
    Tensor được stack; các field text/id giữ dạng list.
    """
    output: Dict[str, Any] = {}

    for key in batch[0]:
        values = [item[key] for item in batch]
        if torch.is_tensor(values[0]):
            output[key] = torch.stack(values)
        else:
            output[key] = values

    return output


def build_dataloader(
    data_path: Union[str, Path, Sequence[Union[str, Path]]],
    vocab,
    batch_size: int = 16,
    max_length: int = 256,
    shuffle: bool = False,
    num_workers: int = 0,
    return_text: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = ABSATargetedDataset(
        data_path=data_path,
        vocab=vocab,
        max_length=max_length,
        return_text=return_text,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=absa_collate_fn,
    )


ViTASA_Dataset = ABSATargetedDataset
