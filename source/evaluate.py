import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import transformers
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

transformers.logging.set_verbosity_error()

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from vocabulary import Vocabulary
from dataloader import absa_collate_fn
from joint_decoding import build_label_indices, decode_structured_joint
from model.targeted_absa import TargetedABSAModel


ENCODER_MAP = {
    "phobert_base": "vinai/phobert-base",
    "phobert_large": "vinai/phobert-large",
    "xlm_roberta_base": "xlm-roberta-base",
    "xlm_roberta_large": "xlm-roberta-large",
}

TOKENIZER_MAP = {
    "phobert_base": "vinai/phobert-base",
    "phobert_large": "vinai/phobert-base",
    "xlm_roberta_base": "xlm-roberta-base",
    "xlm_roberta_large": "xlm-roberta-large",
}

DEFAULT_FALLBACK_RUN_NAME = "phobert_large_target_attn_weighted"


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return PROJECT_ROOT / candidate


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def flatten_runs(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    runs: Dict[str, Dict[str, Any]] = {}
    for group_name, group in config.get("experiment_groups", {}).items():
        for run in group.get("runs", []):
            run = dict(run)
            run["group_name"] = group_name
            runs[run["name"]] = run
    return runs


def find_checkpoints(checkpoint_dir: Path) -> List[Path]:
    if checkpoint_dir.is_file():
        return [checkpoint_dir]
    if not checkpoint_dir.exists():
        return []
    return sorted(checkpoint_dir.rglob("best_model.pt"))


def infer_run_name(checkpoint_path: Path, runs: Dict[str, Dict[str, Any]]) -> Optional[str]:
    for part in reversed(checkpoint_path.parts):
        if part in runs:
            return part

    if checkpoint_path.name != "best_model.pt" and checkpoint_path.stem in runs:
        return checkpoint_path.stem

    return None


def choose_fallback_run_name(config: Dict[str, Any], runs: Dict[str, Dict[str, Any]]) -> str:
    if DEFAULT_FALLBACK_RUN_NAME in runs:
        return DEFAULT_FALLBACK_RUN_NAME

    for run_name in config.get("recommended_first_runs", []):
        if run_name in runs:
            return run_name

    if not runs:
        raise ValueError("Không tìm thấy run nào trong experiments.yaml")

    return next(iter(runs))


def load_state_dict(checkpoint_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint không đúng định dạng state_dict: {checkpoint_path}")

    if any(str(k).startswith("module.") for k in checkpoint.keys()):
        checkpoint = {str(k).replace("module.", "", 1): v for k, v in checkpoint.items()}

    return checkpoint


def forward_pass(model: torch.nn.Module, batch: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    target_mask = batch.get("target_mask")
    sentence_mask = batch.get("sentence_mask")
    token_type_ids = batch.get("token_type_ids")
    aspect_ids = batch.get("aspect_id")

    if target_mask is not None:
        target_mask = target_mask.to(device)
    if sentence_mask is not None:
        sentence_mask = sentence_mask.to(device)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)
    if aspect_ids is not None:
        aspect_ids = aspect_ids.to(device)

    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        target_mask=target_mask,
        sentence_mask=sentence_mask,
        aspect_ids=aspect_ids,
    )


def decode_joint_from_label_ids(vocab: Vocabulary, label_ids: List[int]) -> Tuple[List[int], List[int], List[str]]:
    aspect_ids: List[int] = []
    sentiment_ids: List[int] = []
    joint_labels: List[str] = []

    for label_id in label_ids:
        label = vocab.id2label[int(label_id)]
        aspect, sentiment = vocab.split_label(label)
        aspect_id = vocab.encode_aspect(aspect)
        sentiment_id = vocab.encode_sentiment(sentiment)
        aspect_ids.append(aspect_id)
        sentiment_ids.append(sentiment_id)
        joint_labels.append(f"{aspect}#{sentiment}")

    return aspect_ids, sentiment_ids, joint_labels


class ABSAEvaluationDataset(Dataset):
    """
    Dataset riêng cho evaluation.

    Khác dataloader train: không yêu cầu encode full label_id, vì một số cặp
    aspect-sentiment có thể chỉ xuất hiện ở test. Multitask model vẫn đánh giá
    được nếu aspect và sentiment riêng lẻ đã nằm trong vocab train/dev.
    """

    def __init__(self, data_path: Path, vocab: Vocabulary, max_length: int = 128):
        self.vocab = vocab
        self.max_length = max_length
        self.samples: List[Dict[str, Any]] = []
        self.stats = {
            "total_annotations": 0,
            "kept_annotations": 0,
            "skipped_annotations": 0,
            "unseen_supported_full_labels": set(),
            "skipped_full_labels": set(),
            "missing_aspects": set(),
            "missing_sentiments": set(),
        }
        self._build_samples(data_path)

    def _build_samples(self, data_path: Path) -> None:
        with data_path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                item = json.loads(line)
                text = item.get("data", "")
                if not isinstance(text, str):
                    continue

                for ann in self.vocab.get_annotations_from_sample(item):
                    self.stats["total_annotations"] += 1
                    if not ann["target"]:
                        self.stats["skipped_annotations"] += 1
                        continue

                    label = ann["label"]
                    aspect = ann["aspect"]
                    sentiment = ann["sentiment"]

                    try:
                        aspect_id = self.vocab.encode_aspect(aspect)
                    except KeyError:
                        self.stats["skipped_full_labels"].add(label)
                        self.stats["missing_aspects"].add(aspect)
                        self.stats["skipped_annotations"] += 1
                        continue

                    try:
                        sentiment_id = self.vocab.encode_sentiment(sentiment)
                    except KeyError:
                        self.stats["skipped_full_labels"].add(label)
                        self.stats["missing_sentiments"].add(sentiment)
                        self.stats["skipped_annotations"] += 1
                        continue

                    if label not in self.vocab.label2id:
                        self.stats["unseen_supported_full_labels"].add(label)

                    self.samples.append({
                        "text": text,
                        "target": ann["target"],
                        "aspect_id": aspect_id,
                        "sentiment_id": sentiment_id,
                        "joint_target": f"{aspect}#{sentiment}",
                    })
                    self.stats["kept_annotations"] += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        encoded = self.vocab.tokenize_text_pair(
            sample["text"],
            sample["target"],
            max_length=self.max_length,
        )
        return {
            **encoded,
            "aspect_id": torch.tensor(sample["aspect_id"], dtype=torch.long),
            "sentiment_id": torch.tensor(sample["sentiment_id"], dtype=torch.long),
            "joint_target": sample["joint_target"],
        }


def build_eval_dataloader(
    test_data: Path,
    vocab: Vocabulary,
    batch_size: int,
    max_length: int,
) -> Tuple[DataLoader, Dict[str, Any]]:
    dataset = ABSAEvaluationDataset(test_data, vocab, max_length=max_length)
    if len(dataset) == 0:
        raise ValueError("Không có annotation test nào evaluate được với vocab của checkpoint.")

    printable_stats = dict(dataset.stats)
    printable_stats["unseen_supported_full_labels"] = sorted(printable_stats["unseen_supported_full_labels"])
    printable_stats["skipped_full_labels"] = sorted(printable_stats["skipped_full_labels"])
    printable_stats["missing_aspects"] = sorted(printable_stats["missing_aspects"])
    printable_stats["missing_sentiments"] = sorted(printable_stats["missing_sentiments"])

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=absa_collate_fn,
    ), printable_stats


def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    vocab: Vocabulary,
    device: torch.device,
    constrained_joint_decoding: bool = False,
) -> Dict[str, float]:
    model.eval()

    aspect_preds: List[int] = []
    aspect_targets: List[int] = []
    sentiment_preds: List[int] = []
    sentiment_targets: List[int] = []
    joint_preds: List[str] = []
    joint_targets: List[str] = []
    label_head_preds: List[str] = []
    label_aspect_ids, label_sentiment_ids = build_label_indices(vocab, device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            outputs = forward_pass(model, batch, device)

            target_aspects = batch["aspect_id"].cpu().tolist()
            target_sentiments = batch["sentiment_id"].cpu().tolist()
            target_joint_labels = batch["joint_target"]

            aspect_targets.extend(target_aspects)
            sentiment_targets.extend(target_sentiments)
            joint_targets.extend(target_joint_labels)

            if "aspect_logits" in outputs and "sentiment_logits" in outputs:
                if constrained_joint_decoding:
                    aspect_tensor, sentiment_tensor, label_tensor = decode_structured_joint(
                        outputs["aspect_logits"],
                        outputs["sentiment_logits"],
                        label_aspect_ids,
                        label_sentiment_ids,
                    )
                    batch_aspect_preds = aspect_tensor.cpu().tolist()
                    batch_sentiment_preds = sentiment_tensor.cpu().tolist()
                    batch_joint_preds = [
                        vocab.id2label[int(label_id)]
                        for label_id in label_tensor.cpu().tolist()
                    ]
                else:
                    batch_aspect_preds = torch.argmax(outputs["aspect_logits"], dim=-1).cpu().tolist()
                    batch_sentiment_preds = torch.argmax(outputs["sentiment_logits"], dim=-1).cpu().tolist()
                    batch_joint_preds = [
                        f"{vocab.id2aspect[int(aspect_id)]}#{vocab.id2sentiment[int(sentiment_id)]}"
                        for aspect_id, sentiment_id in zip(batch_aspect_preds, batch_sentiment_preds)
                    ]

                aspect_preds.extend(batch_aspect_preds)
                sentiment_preds.extend(batch_sentiment_preds)
                joint_preds.extend(batch_joint_preds)

            elif "label_logits" in outputs:
                label_ids = torch.argmax(outputs["label_logits"], dim=-1).cpu().tolist()
                batch_aspect_preds, batch_sentiment_preds, batch_joint_preds = decode_joint_from_label_ids(vocab, label_ids)

                aspect_preds.extend(batch_aspect_preds)
                sentiment_preds.extend(batch_sentiment_preds)
                joint_preds.extend(batch_joint_preds)

            if "label_logits" in outputs:
                label_ids = torch.argmax(outputs["label_logits"], dim=-1).cpu().tolist()
                _, _, batch_label_head_preds = decode_joint_from_label_ids(vocab, label_ids)
                label_head_preds.extend(batch_label_head_preds)

    metrics = {
        "aspect_macro_f1": f1_score(aspect_targets, aspect_preds, average="macro", zero_division=0),
        "sentiment_macro_f1": f1_score(sentiment_targets, sentiment_preds, average="macro", zero_division=0),
        "joint_macro_f1": f1_score(joint_targets, joint_preds, average="macro", zero_division=0),
        "aspect_accuracy": accuracy_score(aspect_targets, aspect_preds),
        "sentiment_accuracy": accuracy_score(sentiment_targets, sentiment_preds),
        "joint_accuracy": accuracy_score(joint_targets, joint_preds),
    }

    if label_head_preds:
        metrics["label_head_joint_macro_f1"] = f1_score(
            joint_targets,
            label_head_preds,
            average="macro",
            zero_division=0,
        )
        metrics["label_head_joint_accuracy"] = accuracy_score(joint_targets, label_head_preds)

    return metrics


def build_model_from_run(run: Dict[str, Any], vocab: Vocabulary) -> TargetedABSAModel:
    encoder_key = run.get("encoder_key", "phobert_base")
    encoder_name = ENCODER_MAP.get(encoder_key, "vinai/phobert-base")

    return TargetedABSAModel(
        encoder_name=encoder_name,
        num_aspects=vocab.num_aspects,
        num_sentiments=vocab.num_sentiments,
        num_labels=vocab.num_labels,
        label_structure_key=run.get("label_structure_key", "multitask_aspect_sentiment"),
        attention_key=run.get("attention_key", "cls_pooling"),
        dropout=float(run.get("dropout", 0.1)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate targeted ABSA checkpoints on Data/test.jsonl")
    parser.add_argument("--domain", choices=["hotel", "restaurant", "combined"], default="hotel")
    parser.add_argument("--config", default="config/experiments.yaml")
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--dev-data", default=None)
    parser.add_argument("--test-data", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--checkpoint", default=None, help="Đường dẫn trực tiếp tới file .pt nếu chỉ muốn evaluate một checkpoint.")
    parser.add_argument("--run-name", default=None, help="Tên run trong experiments.yaml nếu chỉ muốn evaluate một run.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output", default="reports/test_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = resolve_path(args.config)
    domain_prefix = "" if args.domain == "combined" else f"{args.domain}_"
    train_data = resolve_path(args.train_data or f"Data/{domain_prefix}train.jsonl")
    dev_data = resolve_path(args.dev_data or f"Data/{domain_prefix}dev.jsonl")
    test_data = resolve_path(args.test_data or f"Data/{domain_prefix}test.jsonl")
    checkpoint_dir = resolve_path(args.checkpoint_dir)
    if args.checkpoint_dir == "checkpoints" and not checkpoint_dir.exists():
        fallback_checkpoint_dir = PROJECT_ROOT / "checkpoint"
        if fallback_checkpoint_dir.exists():
            checkpoint_dir = fallback_checkpoint_dir
    if args.checkpoint_dir == "checkpoints":
        domain_checkpoint_dir = checkpoint_dir / args.domain
        if domain_checkpoint_dir.exists():
            checkpoint_dir = domain_checkpoint_dir
    output_path = resolve_path(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}")

    config = load_yaml(config_path)
    runs = flatten_runs(config)

    if args.checkpoint:
        checkpoints = [resolve_path(args.checkpoint)]
    else:
        checkpoints = find_checkpoints(checkpoint_dir)

    if args.run_name:
        checkpoints = [path for path in checkpoints if infer_run_name(path, runs) in {args.run_name, None}]

    if not checkpoints:
        expected = checkpoint_dir / "*" / "*" / "best_model.pt"
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint để evaluate. Kiểm tra lại --checkpoint hoặc thư mục: {expected}"
        )

    all_results: List[Dict[str, Any]] = []
    fallback_run_name = choose_fallback_run_name(config, runs)

    for checkpoint_path in checkpoints:
        inferred_run_name = infer_run_name(checkpoint_path, runs)
        run_name = args.run_name or inferred_run_name or fallback_run_name
        if inferred_run_name is None and args.run_name is None:
            print(
                f"Không suy ra được run từ checkpoint {checkpoint_path}. "
                f"Tạm dùng run mặc định: {run_name}. "
                "Nếu checkpoint thuộc run khác, hãy truyền --run-name."
            )

        if run_name not in runs:
            raise ValueError(f"Run '{run_name}' không có trong {config_path}")

        run = runs[run_name]
        encoder_key = run.get("encoder_key", "phobert_base")
        encoder_name = ENCODER_MAP.get(encoder_key, "vinai/phobert-base")
        tokenizer_name = TOKENIZER_MAP.get(encoder_key, encoder_name)
        vocab = Vocabulary(
            data_path=[str(train_data), str(dev_data)],
            model_name=tokenizer_name,
        )
        test_loader, test_stats = build_eval_dataloader(
            test_data,
            vocab,
            batch_size=args.batch_size,
            max_length=int(run.get("max_length", args.max_length)),
        )

        print("=" * 80)
        print(f"RUN: {run_name}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Encoder: {run.get('encoder_key')} | Attention: {run.get('attention_key')} | Label: {run.get('label_structure_key')}")
        print(
            "Test annotations | "
            f"kept: {test_stats['kept_annotations']}/{test_stats['total_annotations']} | "
            f"skipped: {test_stats['skipped_annotations']}"
        )
        if test_stats["unseen_supported_full_labels"]:
            print(
                "Full-label mới trong test vẫn được giữ nếu aspect/sentiment đã biết: "
                f"{test_stats['unseen_supported_full_labels']}"
            )
        if test_stats["missing_aspects"] or test_stats["missing_sentiments"]:
            print(
                "Bỏ qua annotation có aspect/sentiment ngoài vocab checkpoint: "
                f"labels={test_stats['skipped_full_labels']}, "
                f"missing_aspects={test_stats['missing_aspects']}, "
                f"missing_sentiments={test_stats['missing_sentiments']}"
            )

        model = build_model_from_run(run, vocab).to(device)
        state_dict = load_state_dict(checkpoint_path, device)
        model.load_state_dict(state_dict, strict=True)

        metrics = evaluate_model(
            model,
            test_loader,
            vocab,
            device,
            constrained_joint_decoding=bool(run.get("constrained_joint_decoding", False)),
        )
        result = {
            "run_name": run_name,
            "checkpoint": str(checkpoint_path),
            "test_stats": test_stats,
            "metrics": metrics,
        }
        all_results.append(result)

        print(
            "Test | "
            f"Joint Macro-F1: {metrics['joint_macro_f1'] * 100:.2f}% | "
            f"Aspect Macro-F1: {metrics['aspect_macro_f1'] * 100:.2f}% | "
            f"Sentiment Macro-F1: {metrics['sentiment_macro_f1'] * 100:.2f}% | "
            f"Joint Acc: {metrics['joint_accuracy'] * 100:.2f}%"
        )
        if "label_head_joint_macro_f1" in metrics:
            print(f"Label-head Joint Macro-F1: {metrics['label_head_joint_macro_f1'] * 100:.2f}%")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"Đã lưu kết quả tại: {output_path}")


if __name__ == "__main__":
    main()
