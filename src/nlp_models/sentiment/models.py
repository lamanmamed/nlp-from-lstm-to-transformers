"""DistilBERT models used for aspect sentiment classification."""

from __future__ import annotations

import torch
from torch import nn

from .pooling import masked_mean


class MeanPooledDistilBertClassifier(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", hidden_size: int = 16, num_labels: int = 3):
        super().__init__()
        from transformers import DistilBertModel

        self.encoder = DistilBertModel.from_pretrained(model_name)
        self.hidden = nn.Linear(self.encoder.config.hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = masked_mean(encoded, attention_mask)
        return self.output(torch.sigmoid(self.hidden(pooled)))


class DistilBertLSTMClassifier(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", lstm_hidden_size: int = 100, num_labels: int = 3):
        super().__init__()
        from transformers import DistilBertModel

        self.encoder = DistilBertModel.from_pretrained(model_name)
        self.lstm = nn.LSTM(
            self.encoder.config.hidden_size,
            lstm_hidden_size,
            batch_first=True,
        )
        self.output = nn.Linear(lstm_hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lengths = attention_mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            encoded, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.output(hidden[-1])
