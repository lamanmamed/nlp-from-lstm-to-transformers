"""Generate a summary with a pretrained or locally fine-tuned T5 checkpoint."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from nlp_models.summarization.preprocessing import add_summarization_prefix


parser = argparse.ArgumentParser()
parser.add_argument("text", help="Document to summarize")
parser.add_argument("--model", default="t5-small")
parser.add_argument("--max-input-length", type=int, default=1024)
parser.add_argument("--max-output-length", type=int, default=128)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

inputs = tokenizer(
    add_summarization_prefix(args.text),
    return_tensors="pt",
    max_length=args.max_input_length,
    truncation=True,
).to(device)
output = model.generate(**inputs, max_length=args.max_output_length)
print(tokenizer.decode(output[0], skip_special_tokens=True))
