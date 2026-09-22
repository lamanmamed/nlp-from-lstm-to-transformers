"""DistilBERT models for predicting humour and offense ratings from social-media text."""

from __future__ import annotations

import torch
from torch import nn

from nlp_models.sentiment.pooling import masked_mean


class DistilBertHumourRegressor(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", hidden_size: int = 16):
        super().__init__()
        from transformers import DistilBertModel

        self.encoder = DistilBertModel.from_pretrained(model_name)
        self.hidden = nn.Linear(self.encoder.config.hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = masked_mean(encoded, attention_mask)
        return self.output(torch.sigmoid(self.hidden(pooled)))


class DistilBertHumourOffenseRegressor(nn.Module):
    """Shared DistilBERT representation with separate humour and offense outputs."""

    def __init__(self, model_name: str = "distilbert-base-uncased", hidden_size: int = 16):
        super().__init__()
        from transformers import DistilBertModel

        self.encoder = DistilBertModel.from_pretrained(model_name)
        self.hidden = nn.Linear(self.encoder.config.hidden_size, hidden_size)
        self.humour_output = nn.Linear(hidden_size, 1)
        self.offense_output = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = masked_mean(encoded, attention_mask)
        shared = torch.sigmoid(self.hidden(pooled))
        return torch.cat([
            self.humour_output(shared),
            self.offense_output(shared),
        ], dim=1)
