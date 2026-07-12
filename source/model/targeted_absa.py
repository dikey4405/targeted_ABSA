import torch
from torch import nn
from transformers import AutoModel
from .attn_pooling import AttentionPooling
from .conditional_attn import ConditionalAttention
from .gated_fusion import GatedFusion
from .mlp_head import MLPHead

class TargetedABSAModel(nn.Module):
    SUPPORTED_ATTENTION_KEYS = {
        "cls_pooling",
        "mean_pooling",
        "attention_pooling",
        "cls_attention_gated_fusion",
        "target_conditioned_attention",
        "aspect_conditioned_attention",
        "cascaded_attention",
        "multihead_pooling",
    }

    SUPPORTED_LABEL_STRUCTURES = {
        "joint_label",
        "multitask_aspect_sentiment",
        "multitask_with_joint_aux",
        "aspect_binary_plus_sentiment",
        "aspect_sentiment_vector_3",
        "aspect_sentiment_vector_4",
    }

    def __init__(
        self,
        encoder_name: str,
        num_aspects: int,
        num_sentiments: int,
        num_labels: int = None,
        label_structure_key: str = "multitask_aspect_sentiment",
        attention_key: str = "cls_pooling",
        dropout: float = 0.2,
        classifier_hidden_size: int = 256,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        if attention_key not in self.SUPPORTED_ATTENTION_KEYS:
            raise ValueError(f"Unsupported attention_key: {attention_key}")

        if label_structure_key not in self.SUPPORTED_LABEL_STRUCTURES:
            raise ValueError(f"Unsupported label_structure_key: {label_structure_key}")

        if label_structure_key in {"joint_label", "multitask_with_joint_aux"} and num_labels is None:
            raise ValueError(f"num_labels is required for label_structure_key={label_structure_key}")

        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = self.encoder.config.hidden_size

        self.num_aspects = num_aspects
        self.num_sentiments = num_sentiments
        self.label_structure_key = label_structure_key
        self.attention_key = attention_key

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.attn_pool = AttentionPooling(
            hidden_size=self.hidden_size,
            attn_size=classifier_hidden_size,
            dropout=dropout,
        )
        self.cond_attn = ConditionalAttention(
            hidden_size=self.hidden_size,
            attn_size=classifier_hidden_size,
        )
        self.gated_fusion = GatedFusion(self.hidden_size)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.aspect_embedding = nn.Embedding(num_aspects, self.hidden_size)

        self.aspect_head = MLPHead(
            self.hidden_size,
            num_aspects,
            dropout=dropout,
            mid_size=classifier_hidden_size,
        )

        self.sentiment_head = MLPHead(
            self.hidden_size,
            num_sentiments,
            dropout=dropout,
            mid_size=classifier_hidden_size,
        )

        self.joint_head = None
        if num_labels is not None:
            self.joint_head = MLPHead(
                self.hidden_size,
                num_labels,
                dropout=dropout,
                mid_size=classifier_hidden_size,
            )

        self.grid_head = None
        if label_structure_key == "aspect_sentiment_vector_3":
            self.grid_slots = num_sentiments
            self.grid_head = nn.Linear(self.hidden_size, num_aspects * self.grid_slots)
        elif label_structure_key == "aspect_sentiment_vector_4":
            self.grid_slots = num_sentiments + 1
            self.grid_head = nn.Linear(self.hidden_size, num_aspects * self.grid_slots)
        else:
            self.grid_slots = None

    def mean_pooling(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def masked_mean_pooling(self, hidden_states, mask):
        mask = mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def target_pooling(self, hidden_states, target_mask, fallback):
        """Pool target tokens and use CLS when a truncated target has no token."""
        target_vec = self.masked_mean_pooling(hidden_states, target_mask)
        has_target = target_mask.any(dim=1, keepdim=True)
        return torch.where(has_target, target_vec, fallback)

    def build_representation(
        self,
        hidden_states,
        attention_mask,
        target_mask=None,
        sentence_mask=None,
        aspect_ids=None,
    ):
        cls_vec = hidden_states[:, 0]

        if self.attention_key == "cls_pooling":
            return cls_vec

        if self.attention_key == "mean_pooling":
            return self.mean_pooling(hidden_states, attention_mask)

        if self.attention_key == "attention_pooling":
            return self.attn_pool(hidden_states, attention_mask)

        if self.attention_key == "cls_attention_gated_fusion":
            attn_vec = self.attn_pool(hidden_states, attention_mask)
            return self.gated_fusion(cls_vec, attn_vec)

        if self.attention_key == "multihead_pooling":
            key_padding_mask = attention_mask == 0
            attn_output, _ = self.multihead_attn(
                hidden_states,
                hidden_states,
                hidden_states,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            return self.mean_pooling(attn_output, attention_mask)

        if self.attention_key == "target_conditioned_attention":
            if target_mask is None:
                target_vec = cls_vec
            else:
                target_vec = self.target_pooling(hidden_states, target_mask, cls_vec)

            sent_mask = sentence_mask if sentence_mask is not None else attention_mask
            target_context = self.cond_attn(hidden_states, target_vec, sent_mask)
            return self.gated_fusion(cls_vec, target_context)

        if self.attention_key == "aspect_conditioned_attention":
            if aspect_ids is None:
                aspect_probs = torch.sigmoid(self.aspect_head(cls_vec)) 
                aspect_vec_sum = torch.matmul(aspect_probs, self.aspect_embedding.weight)
                denom = aspect_probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                aspect_vec = aspect_vec_sum / denom
                
            else:
                aspect_vec = self.aspect_embedding(aspect_ids)

            sent_mask = sentence_mask if sentence_mask is not None else attention_mask
            return self.cond_attn(hidden_states, aspect_vec, sent_mask)

        if self.attention_key == "cascaded_attention":
            if target_mask is None:
                target_vec = cls_vec
            else:
                target_vec = self.target_pooling(hidden_states, target_mask, cls_vec)

            sent_mask = sentence_mask if sentence_mask is not None else attention_mask
            aspect_context = self.cond_attn(hidden_states, target_vec, sent_mask)
            sentiment_context = self.cond_attn(hidden_states, aspect_context, sent_mask)
            return self.gated_fusion(cls_vec, sentiment_context)

        raise ValueError(f"Unsupported attention_key: {self.attention_key}")

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        target_mask=None,
        sentence_mask=None,
        aspect_ids=None,
    ):
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        hidden_states = outputs.last_hidden_state

        pooled = self.build_representation(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            target_mask=target_mask,
            sentence_mask=sentence_mask,
            aspect_ids=aspect_ids,
        )

        result = {
            "pooled_output": pooled,
        }

        if self.label_structure_key in [
            "multitask_aspect_sentiment",
            "multitask_with_joint_aux",
            "aspect_binary_plus_sentiment",
        ]:
            result["aspect_logits"] = self.aspect_head(pooled)
            result["sentiment_logits"] = self.sentiment_head(pooled)

        if self.label_structure_key == "joint_label":
            result["label_logits"] = self.joint_head(pooled)

        if self.label_structure_key == "multitask_with_joint_aux":
            result["label_logits"] = self.joint_head(pooled)

        if self.label_structure_key in ["aspect_sentiment_vector_3", "aspect_sentiment_vector_4"]:
            grid_logits = self.grid_head(pooled)
            result["grid_logits"] = grid_logits.view(
                input_ids.size(0),
                self.num_aspects,
                self.grid_slots,
            )

        return result
