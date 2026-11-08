"""
Transformer-based Topic Classifier for Vietnamese Social Listening.
Supports ViSoBERT, PhoBERT, Mean/CLS/Attention Pooling, and Multi-Sample Dropout.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModel, AutoConfig

class AttentionPooling(nn.Module):
    """
    Learned attention pooling layer over token sequence representations.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # last_hidden_state: (B, L, D), attention_mask: (B, L)
        scores = self.attention(last_hidden_state).squeeze(-1) # (B, L)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1) # (B, L, 1)
        pooled = torch.sum(last_hidden_state * weights, dim=1) # (B, D)
        return pooled

class TopicTransformerClassifier(nn.Module):
    """
    Topic Classification Architecture on top of pretrained Vietnamese Transformers.
    """
    def __init__(
        self,
        model_name: str = "bavest/visobert-base-v2",
        num_classes: int = 13,
        pooling: str = "mean",
        use_multi_sample_dropout: bool = True,
        num_dropout_samples: int = 5,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pooling_type = pooling.lower()
        self.use_msd = use_multi_sample_dropout
        self.num_dropout_samples = num_dropout_samples
        
        # Load Pretrained Backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size

        # Pooling mechanism
        if self.pooling_type == "attention":
            self.pooler = AttentionPooling(hidden_size)
        else:
            self.pooler = None

        # Dropout mechanism
        if self.use_msd:
            self.dropouts = nn.ModuleList([
                nn.Dropout(dropout_rate) for _ in range(num_dropout_samples)
            ])
        else:
            self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate)])

        # Linear Classification Head
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Learnable Temperature parameter for post-calibration (initialized to 1.0)
        self.temperature = nn.Parameter(torch.ones(1))

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling_type == "cls":
            return last_hidden_state[:, 0, :]
        elif self.pooling_type == "attention":
            return self.pooler(last_hidden_state, attention_mask)
        else:
            # Mean pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and "token_type_ids" in self.encoder.forward.__code__.co_varnames:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)
        pooled = self._pool(outputs.last_hidden_state, attention_mask)

        # Multi-Sample Dropout forward
        logits_list = [self.classifier(dropout(pooled)) for dropout in self.dropouts]
        # Average logits across dropout samples
        logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)
        return logits

    def predict_probabilities(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        apply_temperature: bool = True
    ) -> torch.Tensor:
        """
        Computes calibrated Softmax probabilities.
        """
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        if apply_temperature and self.temperature.item() > 0:
            logits = logits / self.temperature
        return torch.softmax(logits, dim=-1)
