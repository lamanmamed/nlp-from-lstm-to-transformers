"""LSTM encoder-decoder translation model with optional attention and bidirectionality."""

from __future__ import annotations

import torch
from torch import nn

from .attention import LuongAttention


class Seq2SeqTranslator(nn.Module):
    """Sequence-to-sequence LSTM model matching the architecture used in the project."""

    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        *,
        source_pad_id: int = 0,
        target_pad_id: int = 0,
        embedding_size: int = 100,
        hidden_size: int = 200,
        dropout: float = 0.2,
        use_attention: bool = True,
        bidirectional_encoder: bool = False,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.bidirectional_encoder = bidirectional_encoder
        self.hidden_size = hidden_size

        self.source_embedding = nn.Embedding(
            source_vocab_size, embedding_size, padding_idx=source_pad_id
        )
        self.target_embedding = nn.Embedding(
            target_vocab_size, embedding_size, padding_idx=target_pad_id
        )
        self.embedding_dropout = nn.Dropout(dropout)

        self.encoder = nn.LSTM(
            embedding_size,
            hidden_size,
            batch_first=True,
            bidirectional=bidirectional_encoder,
        )
        self.decoder = nn.LSTM(
            embedding_size,
            hidden_size,
            batch_first=True,
        )

        if bidirectional_encoder:
            self.hidden_projection = nn.Linear(hidden_size * 2, hidden_size)
            self.cell_projection = nn.Linear(hidden_size * 2, hidden_size)
            self.encoder_output_projection = nn.Linear(hidden_size * 2, hidden_size)

        self.attention = LuongAttention() if use_attention else None
        projection_input = hidden_size * 2 if use_attention else hidden_size
        self.output_projection = nn.Linear(projection_input, target_vocab_size)

    def encode(self, source_ids: torch.Tensor):
        embedded = self.embedding_dropout(self.source_embedding(source_ids))
        encoder_outputs, (hidden, cell) = self.encoder(embedded)

        if self.bidirectional_encoder:
            hidden = self.hidden_projection(
                torch.cat([hidden[0], hidden[1]], dim=-1)
            ).unsqueeze(0)
            cell = self.cell_projection(
                torch.cat([cell[0], cell[1]], dim=-1)
            ).unsqueeze(0)
            encoder_outputs = self.encoder_output_projection(encoder_outputs)

        return encoder_outputs, (hidden, cell)

    def forward(self, source_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        encoder_outputs, decoder_state = self.encode(source_ids)
        target_embeddings = self.embedding_dropout(self.target_embedding(target_ids))
        decoder_outputs, _ = self.decoder(target_embeddings, decoder_state)

        if self.attention is not None:
            source_mask = source_ids.ne(self.source_embedding.padding_idx)
            decoder_outputs = self.attention(
                encoder_outputs, decoder_outputs, source_mask=source_mask
            )

        return self.output_projection(decoder_outputs)

    def decode_step(
        self,
        token_ids: torch.Tensor,
        decoder_state,
        encoder_outputs: torch.Tensor,
        source_mask: torch.Tensor | None = None,
    ):
        embedded = self.target_embedding(token_ids)
        decoder_outputs, decoder_state = self.decoder(embedded, decoder_state)

        if self.attention is not None:
            decoder_outputs = self.attention(
                encoder_outputs, decoder_outputs, source_mask=source_mask
            )

        logits = self.output_projection(decoder_outputs)
        return logits, decoder_state
