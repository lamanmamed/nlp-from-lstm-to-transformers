# NLP from LSTMs to Transformers

This repository contains five NLP projects: translating Vietnamese into English, identifying sentiment about a specific part of a restaurant review, generating text with GPT-2, predicting how funny a social-media post is, and summarizing news articles with T5.

The projects use both recurrent neural networks and transformer models. Some parts use pretrained models such as DistilBERT, GPT-2, and T5. Other parts implement important pieces directly, including attention for translation, an LSTM classifier, greedy search, beam search, and repetition control during text generation.

## Translating Vietnamese into English

I trained an LSTM encoder-decoder on Vietnamese-English sentence pairs and compared four versions of the model.

The main question was whether attention helps translation, and whether making the encoder read the Vietnamese sentence in both directions helps further.

| Encoder | Attention | Test BLEU |
| --- | --- | ---: |
| Unidirectional | No | 3.68 |
| Unidirectional | Yes | **15.01** |
| Bidirectional | No | 3.25 |
| Bidirectional | Yes | 13.66 |

Adding attention made the biggest difference. Without attention, the decoder mainly depends on the final state produced after the encoder has read the whole Vietnamese sentence. With attention, the decoder can look back at different words in the source sentence while generating each English word.

The bidirectional encoder did not improve the result in these runs. I combined the forward and backward encoder states and projected them back to the size expected by the decoder, but both bidirectional versions still scored below the corresponding unidirectional models.

The written results also contain an earlier run with BLEU 4.11 without attention and 15.73 with attention. I use the values above for the main comparison because all four model versions were evaluated together in the same notebook run.

## Identifying sentiment about food, service, or decor

The sentiment task is more specific than ordinary positive-versus-negative review classification.

Each example contains:

- a restaurant review
- an aspect such as `food`, `decor`, or `service`
- the sentiment toward that particular aspect

The model therefore receives both pieces of text:

```text
review text [SEP] aspect
```

For example, the same review could be positive about the food but negative about the service.

I compared three DistilBERT-based models:

| Model | Test accuracy |
| --- | ---: |
| DistilBERT sequence classifier | 82.04% |
| DistilBERT + mean-pooled token features + MLP | 80.54% |
| DistilBERT token features + LSTM | **83.08%** |

The strongest saved result was **83.08%**. In that version, DistilBERT produces a representation for every token in the input, and an LSTM reads those token representations before making the final sentiment prediction.

## Generating text with GPT-2

I implemented two ways of choosing the next tokens from GPT-2 myself rather than relying only on Hugging Face's built-in `generate()` function.

**Greedy search** always chooses the single most likely next token.

**Beam search** keeps several possible continuations at the same time and continues expanding the most promising ones. I also added a version that prevents recently generated tokens from being selected again too soon.

For the prompt:

```text
The cat slept on the
```

the saved run produced:

| Method | Cumulative log probability |
| --- | ---: |
| Greedy search | -62.5860 |
| Beam search, width 3 | **-46.0669** |
| Beam search with recent-token blocking | -77.4257 |

The ordinary beam search found a sequence that GPT-2 considered more likely than the greedy sequence, but it also became repetitive. One generated continuation repeated phrases such as `It was like a nightmare`.

Blocking recently used tokens reduced that repetition. The trade-off was that GPT-2 assigned the resulting sequence a lower probability.

I also tested random sampling, different temperature values, and top-p sampling using Hugging Face's generation functions. Higher temperature made the output more random, while lower temperature made it more repetitive. Top-p sampling restricted each choice to a smaller set of likely next tokens while still allowing some variation.

## Predicting how funny a social-media post is

This task predicts a **number**, not a category. Each post has a humour rating from 0 to 5, and the model predicts that rating from the text.

Before training, the ratings are divided by 5 so the target values fall between 0 and 1. The reported mean squared error values are therefore also based on that 0-1 scale.

The baseline model uses DistilBERT to represent the post, averages the token representations while ignoring padding, and passes the result through a small neural network that outputs one humour score.

I tested several alternatives:

| Setup | Test MSE |
| --- | ---: |
| Baseline DistilBERT regressor | **0.0117** |
| Extra preprocessing for social-media text | 0.0129 |
| Average prediction from three trained models | 0.0124 |
| One model predicting humour and offense together | 0.0128 |

None of these versions beat the baseline in the saved runs.

There is also code for creating extra training examples through synonym replacement and random word deletion. However, the later training cell reloads the original training text instead of the generated examples. Because of that, I do not treat the saved `0.0128` MSE as evidence that data augmentation helped.

## Summarizing news articles with T5

The T5 project uses XSum, a dataset of news articles paired with short summaries.

I fine-tuned T5 to take a full article as input and generate its summary. In the subset used here, the articles averaged about 365 whitespace-separated tokens and the summaries averaged about 21.

The notebook contains a ROUGE comparison for one article:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| Fine-tuned T5 | 0.1200 | 0.0000 | 0.0800 |
| Base T5 | 0.0816 | 0.0426 | 0.0816 |

These numbers are kept as a single-example comparison, not as an overall evaluation of either model, because ROUGE was calculated for only one document.

### Generating text from a small set of words

I also trained T5 on a second task. Instead of giving it a full news article, the input is a random selection of up to 20 words taken from a target summary. The model then tries to reconstruct the full summary.

This is a much less constrained generation task because the selected words do not tell the model their original order or the missing parts of the sentence.

The saved example produces understandable text but repeats some words, including `structural` and `neglected`. I keep that output as an example of the model's behaviour rather than presenting it as a strong result.

## Repository structure

```text
nlp-from-lstm-to-transformers/
├── src/nlp_models/
│   ├── translation/
│   │   ├── attention.py
│   │   ├── data.py
│   │   └── model.py
│   ├── sentiment/
│   │   ├── data.py
│   │   ├── pooling.py
│   │   └── models.py
│   ├── text_generation/
│   │   └── decoding.py
│   ├── social_media/
│   │   ├── evaluation.py
│   │   ├── models.py
│   │   └── preprocessing.py
│   └── summarization/
│       ├── preprocessing.py
│       └── t5_tasks.py
├── scripts/
│   ├── gpt2_text_generation.py
│   ├── t5_summarization_demo.py
│   └── translation_model_shapes.py
├── tests/
├── data/
│   └── README.md
├── pyproject.toml
└── README.md
```

## Running the code

Install the core package:

```bash
pip install -e .
```

Install the extra libraries used for the Hugging Face models and datasets:

```bash
pip install -e ".[transformers]"
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

Run the custom GPT-2 generation code:

```bash
python scripts/gpt2_text_generation.py --prompt "The cat slept on the"
```

The pretrained-model scripts need internet access the first time they run so the model weights and tokenizers can be downloaded.

## Data

The datasets are not included in the repository. `data/README.md` lists the dataset used for each part of the project and how the code expects it to be provided.
