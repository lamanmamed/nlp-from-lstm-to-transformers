"""Compare custom GPT-2 greedy and beam-search generation."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nlp_models.text_generation.decoding import (
    beam_search,
    beam_search_no_repeat,
    greedy_search,
)


parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="The cat slept on the")
parser.add_argument("--model", default="gpt2")
parser.add_argument("--max-length", type=int, default=50)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

for name, fn, kwargs in [
    ("Greedy", greedy_search, {}),
    ("Beam search", beam_search, {"beam_size": 3}),
    (
        "Beam search with recent-token blocking",
        beam_search_no_repeat,
        {"beam_size": 3, "recent_token_window": 20},
    ),
]:
    sequence, score = fn(
        model,
        input_ids,
        max_length=args.max_length,
        eos_token_id=tokenizer.eos_token_id,
        **kwargs,
    )
    print(f"\n{name}\n{'-' * len(name)}")
    print(tokenizer.decode(sequence[0], skip_special_tokens=True))
    print(f"log probability: {score:.4f}")

print("\nSampling with the Hugging Face generation API")
for label, generation_kwargs in [
    ("random", {"do_sample": True}),
    ("high temperature", {"do_sample": True, "temperature": 2.0}),
    ("low temperature", {"do_sample": True, "temperature": 0.2}),
    ("top-p", {"do_sample": True, "top_p": 0.92, "top_k": 0}),
]:
    torch.manual_seed(0)
    output = model.generate(
        input_ids,
        max_length=args.max_length,
        pad_token_id=tokenizer.eos_token_id,
        **generation_kwargs,
    )
    print(f"\n{label}: {tokenizer.decode(output[0], skip_special_tokens=True)}")
