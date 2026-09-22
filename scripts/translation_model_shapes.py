"""Small shape check for the four translation configurations used in the project."""

import torch

from nlp_models.translation.model import Seq2SeqTranslator


source = torch.tensor([[2, 4, 5, 0], [2, 6, 7, 8]])
target = torch.tensor([[1, 3, 4], [1, 5, 6]])

for attention in [False, True]:
    for bidirectional in [False, True]:
        model = Seq2SeqTranslator(
            source_vocab_size=20,
            target_vocab_size=25,
            use_attention=attention,
            bidirectional_encoder=bidirectional,
        )
        logits = model(source, target)
        print(
            f"attention={attention:<5} bidirectional={bidirectional:<5} "
            f"output={tuple(logits.shape)}"
        )
