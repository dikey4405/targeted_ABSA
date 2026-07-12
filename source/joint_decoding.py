from typing import Tuple

import torch


def build_label_indices(vocab, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map each full-label id to its aspect id and sentiment id."""
    aspect_ids = []
    sentiment_ids = []

    for label_id in range(vocab.num_labels):
        aspect, sentiment = vocab.split_label(vocab.id2label[label_id])
        aspect_ids.append(vocab.encode_aspect(aspect))
        sentiment_ids.append(vocab.encode_sentiment(sentiment))

    return (
        torch.tensor(aspect_ids, dtype=torch.long, device=device),
        torch.tensor(sentiment_ids, dtype=torch.long, device=device),
    )


def structured_joint_logits(
    aspect_logits: torch.Tensor,
    sentiment_logits: torch.Tensor,
    label_aspect_ids: torch.Tensor,
    label_sentiment_ids: torch.Tensor,
) -> torch.Tensor:
    """Score only aspect-sentiment pairs represented by the domain vocabulary."""
    return (
        aspect_logits.index_select(1, label_aspect_ids)
        + sentiment_logits.index_select(1, label_sentiment_ids)
    )


def decode_structured_joint(
    aspect_logits: torch.Tensor,
    sentiment_logits: torch.Tensor,
    label_aspect_ids: torch.Tensor,
    label_sentiment_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    joint_logits = structured_joint_logits(
        aspect_logits,
        sentiment_logits,
        label_aspect_ids,
        label_sentiment_ids,
    )
    label_ids = joint_logits.argmax(dim=-1)
    return (
        label_aspect_ids[label_ids],
        label_sentiment_ids[label_ids],
        label_ids,
    )
