# Training

We use [Qwen3](https://github.com/QwenLM/Qwen3) dense models (0.6B / 1.7B / 4B)
as the language-model backbone. The training framework is intentionally
minimal: rather than defining new wrapper modules, we extend the vocabulary and
LM head of the base LLM so that flattened discrete image tokens are modeled
exactly like text tokens, keeping the original LLM loading path intact.

## Model preparation

We extend the base LLM's vocabulary to accommodate image tokens. No new tokens
are added to the *text tokenizer* itself — image tokens are produced by the
image tokenizer (VQGAN-family). We do add three special tokens:

- `<boi>` / `<eoi>` — mark the beginning / end of a block of image tokens.
- `<unconditional>` — enables Classifier-Free Guidance (CFG) during training.
  During T2I training the caption is replaced by `<unconditional>` with
  probability `1 - cfg_ratio`; `--cfg_ratio` defaults to `0.9`, i.e. a 10% drop
  rate.

You can map these onto existing reserved/unused tokens or edit the tokenizer
files directly (both `tokenizer.json` and `tokenizer_config.json`), setting
`"special": true` for the three tokens.

Then extend the embeddings and LM head:

```bash
python liquid/expand_vocabulary.py \
  --model_path '/path/to/Qwen3-0.6B-Base' \
  --save_path '/path/to/Qwen3-0.6B-Base-addtoken' \
  --num_add_token 16384
```

`--num_add_token` must match the image tokenizer's codebook size — 16384 for
GigaTok / GigaTok-DINO, and 1024 / 8192 / 16384 for the corresponding IBQ
variants. The training scripts pass this automatically via `VOCAB`.

`resize_token_embeddings` initializes the new embeddings with the mean of the
existing ones (`mean_resizing`, the default since `transformers>=4.46.0`), which
reduces the KL divergence of the next-token distribution before and after
resizing and yields a lower initial loss.

## Environment

Build the conda environment with the helper script:

```bash
bash scripts/setup_env.sh
conda activate liquid
```

This creates a `python=3.10` environment and installs the package,
`flash-attn==2.5.8`, `transformers==4.51.0`, and the data / logging utilities
(see [scripts/setup_env.sh](scripts/setup_env.sh)). It also builds the `t2v` and
`base` environments used for GenAI-Bench and WISE scoring, and installs the
three extra packages the image tokenizers need — all tokenization runs inside
`liquid`.

`transformers>=4.51.0` is a hard requirement: Qwen3 support first shipped in
4.51.0, so earlier releases cannot load these models at all. Beyond that the
framework is not sensitive to exact versions; newer `torch` / `transformers` /
`flash-attention` releases should also work.

## Continual pretraining

We train the unified autoregressive model on mixed-modal (image-text) and
pure-text data. We apply a Warmup-Stable-Decay (WSD) schedule (0.03 warmup, last
20% linear decay). All model sizes use `lr=3e-5` and a **global batch size of
512** sequences of 2048 tokens.

The scripts hold that global batch on any number of GPUs, at every model size:
gradient accumulation is derived as `GLOBAL_BS / (BS × NPROC × NNODES)`, so the
defaults (`BS=16`, `NPROC=8`, `NNODES=1`) reach 512 on a single node. Override
`BS`, `NPROC`, `NNODES` or `GLOBAL_BS` as needed — the run aborts if they don't
divide evenly rather than training at the wrong batch size. The paper's runs
used 4 nodes × 8 GPUs.

[`scripts/qwen3_0_6b_mixpretrain.sh`](scripts/qwen3_0_6b_mixpretrain.sh) first
runs `expand_vocabulary.py` and then launches distributed training with
`torchrun` (a single 8-GPU node by default). Configure the run through
environment variables:

```bash
# single node
TOKENIZER=ibq_8192 VOCAB=8192 LR=3e-5 bash scripts/qwen3_0_6b_mixpretrain.sh

# multi-node (launch on each node)
NNODES=4 NODE_RANK=$RANK MASTER_ADDR=$HEAD_IP bash scripts/qwen3_0_6b_mixpretrain.sh

# annealing / decay phase (resumes from a stage-1 checkpoint)
STABLE_CKPT=checkpoint-23438 bash scripts/qwen3_0_6b_mixpretrain_anneal.sh
```

`TOKENIZER` selects which pre-tokenized datasets to read (`VOCAB` must match its
codebook size). Set the base model and data roots via `.env` (see
[README](README.md#path-configuration)); pre-tokenize the data with the
tokenizer of interest (see [evaluation/TOKENIZERS.md](evaluation/TOKENIZERS.md)).

The stable phase uses a `constant_with_warmup` schedule; the
[annealing script](scripts/qwen3_0_6b_mixpretrain_anneal.sh) then resumes from
`STABLE_CKPT` in the same output directory and linearly decays the learning rate
(the WSD decay phase), using `rename_ckpt.py` to set the resume point and to mark
the final annealed checkpoint (`acheckpoint-*`).

> **Size the annealing data to the resume step.** The decay is the last 1/5 of
> the whole run, and `--warmup_ratio 0.8` puts the LR peak at the resume point,
> so the annealing run's **total** steps must be `1.25 × (STABLE_CKPT step)`.
> Total steps are set by the data amount, so set `ANNEAL_PERCENTAGE`
> accordingly — e.g. if the stable phase ended at fraction `f` of the data
> budget, size the annealing data up to `1.25·f` of it (keep `--warmup_ratio`
> at `0.8`):
>
> ```bash
> STABLE_CKPT=checkpoint-23438 ANNEAL_PERCENTAGE="0.05555^^0.25^^0.25^^0.14075^^0.25" \
>   bash scripts/qwen3_0_6b_mixpretrain_anneal.sh
> ```
>
> `ANNEAL_PERCENTAGE` has **five** entries, matching `--data_path`: DCLM, LAION,
> JourneyDB, JourneyDB again, BLIP3o. JourneyDB is deliberately listed twice so
> the mixture samples it at ~1.563× the other image-text sources; keep the two
> entries in the same 1 : 0.563 ratio when rescaling.

### Model sizes (0.6B / 1.7B / 4B)

The 1.7B and 4B models use the same recipe; thin wrappers only set the backbone,
so every hyperparameter — including the node count — is identical across sizes:

```bash
bash scripts/qwen3_1_7b_mixpretrain.sh          # 1.7B
bash scripts/qwen3_4b_mixpretrain.sh            # 4B

# annealing and SFT variants exist for each size, e.g.
bash scripts/qwen3_1_7b_mixpretrain_anneal.sh
bash scripts/qwen3_4b_sft.sh
```

All wrappers accept the same environment variables as the 0.6B scripts
(`TOKENIZER`, `VOCAB`, `LR`, `BS`, `STABLE_CKPT`, `NNODES`, ...).

## Supervised finetuning (SFT)

SFT trains for 2 epochs on instruction-following and captioning data with a
cosine schedule (0.03 warmup, `lr=5e-5`, global batch size 1024):

```bash
TOKENIZER=gigatok BASE_CKPT=/path/to/stage1/checkpoint bash scripts/qwen3_0_6b_sft.sh
```

Gradient accumulation is derived the same way as in stage 1, against
`GLOBAL_BS=1024`, so the shipped defaults reach 1024 at every model size.

`BASE_CKPT` is the stage-1 continual-pretraining checkpoint; `TOKENIZER`
selects the SFT data tokenized with the matching tokenizer. See
[scripts/qwen3_0_6b_sft.sh](scripts/qwen3_0_6b_sft.sh) and [Data.md](./Data.md)
for how to prepare the pretraining and SFT data.
