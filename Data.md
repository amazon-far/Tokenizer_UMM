# Data

This document describes the full data pipeline: from raw sources to the
HuggingFace datasets consumed by the trainer. All paths use the environment
variables in `.env` (`$DATA_ROOT`, etc.; see [README](README.md#path-configuration)).
All datasets used are publicly accessible; all images are resized to 256×256 and
pre-tokenized into discrete codes **before** training.

## Overview

Because concatenating HuggingFace datasets requires identical features, every
source is first converted to a **unified JSONL schema** and then packed into a
HuggingFace (Arrow) dataset. One JSON object per line:

```python
# text
{"data_type": "text_pretrain", "text": "..."}
# image-text (image encoded to discrete tokens by the image tokenizer)
{"data_type": "image_text", "source": "laion-aesthetics",
 "text": "a caption", "vqcode_512": "[5795, 2236, ...]",
 "length": 1234, "vqcode_256": "no", "vqcode_multi768": "no"}
```

`sample_laion.jsonl` and `sample_journeydb.jsonl` in the repository root show the
same schema as complete records. **Both files contain synthetic placeholder
data**: the captions were written by hand and the `vqcode_512` values are random
token ids. They document the format only and are not samples of any third-party
dataset. Obtain the actual data from the upstream sources under their own terms.

The trainer receives a `^^`-joined list of dataset paths via `--data_path` and a
matching `--percentage` list (see [TRAIN.md](TRAIN.md)). Each **image tokenizer**
produces its **own** tokenized copies of the image-text datasets, named with a
tokenizer suffix, e.g. `laion_tokenized_<tokenizer>_hf`. Set once:

```bash
export TOKENIZER_NAME=gigatok   # gigatok | gigatok_dino | ibq_1024 | ibq_8192 | ibq_16384 | unitok | unitok_sem
```

The pipeline has four stages:

1. Text data → HF dataset
2. Image-text raw data → parquet
3. Tokenize images → sharded JSONL → HF datasets
4. SFT data

---

## 1. Text data (DataComp-LM)

Convert DCLM shards (`*.jsonl.zst`) into a HuggingFace dataset. Each record is
written with `data_type: "text_pretrain"`, staged as JSONL, then packed to Arrow.

```bash
cd data_process
python convert_DCLM_data.py \
  --input_path  /path/to/raw/DCLM \
  --temp_path   /tmp/dclm_jsonl \
  --save_path   $DATA_ROOT/dclm30m_hf
```

For DCLM stored as WebDataset shards on S3, use the multiprocessing variants
(`convert_DCLM_aws_data*.py`, `convert_DCLM_local_data_multiprocessing.py`) and
`convert_DCLM_local_val_data_multiprocessing.py` for the validation split.

## 2. Image-text raw data → parquet

Each image-text source is first turned into parquet files of image + caption.

- **LAION-Aesthetics** — filter by aesthetic score with
  `laion_aesthetics_v2_filter.py`, then recaption (the paper uses InternVL3-1B;
  external). The result is sharded as `filtered_laion_aesthetics_parquet_recaptioned/part_{1..12}.parquet`
  plus a `val.parquet`.
- **JourneyDB** — recaptioned parquet (`train/`, `validation/`); the paper uses
  GPT-4.5 recaptions (external).
- **BLIP3o** — downloaded from the HuggingFace Hub (e.g.
  `BLIP3o/BLIP3o-Pretrain-Short-Caption`) as tar shards; tokenized **directly**
  (no parquet step) by the per-tokenizer `blip3o_<tok>.py` scripts (see stage 3).
  Two separate sets are used, differing in **scale and role** (not caption
  length): **BLIP3o-short**, a large-scale set (from
  `BLIP3o/BLIP3o-Pretrain-Short-Caption`) used for **pretraining**, and
  **BLIP3o-60k**, a small (~60k) set used for **SFT**. `data_process/blip3o.py`
  is the VQGAN version; `tar_to_parquet.py` / `tar_to_parquet_images.py` are
  helpers. (Note: the internal names `blip3o-60k-short` / `BLIP3o_60k_short`
  refer to BLIP3o-short — it is *not* a 60k-sized set.)

## 3. Tokenize images → sharded JSONL → HF datasets

### 3a. Tokenize (multi-GPU)

The tokenization scripts encode each image to discrete codes with the chosen
tokenizer and write sharded JSONL (`<output_dir>/rank_<r>/batch_*.jsonl`). Each
script spawns one process per visible GPU (`torch.cuda.device_count()` +
`mp.spawn`), so launch it with a single `python` call on an 8-GPU node.

> **All tokenizers run in the `liquid` env** — including the VQGAN / Chameleon
> scripts in `data_process/`. They need only `omegaconf`, `pytorch-lightning`
> and `lightning` on top of it (installed by
> [`scripts/setup_env.sh`](scripts/setup_env.sh); IBQ is the only one that needs
> them). Do **not** install the upstream `requirements.txt` files under
> `evaluation/<tokenizer>/` into `liquid` — they pin older torch / transformers
> builds that would downgrade the training environment.

- The **VQGAN / Chameleon** versions live here: `tokenize_to_jsonl_laion.py`,
  `tokenize_to_jsonl_journeydb.py`.
- The **per-tokenizer** versions live in each tokenizer directory, e.g.
  `evaluation/GigaTok/tokenize_to_jsonl_laion_gigatok.py`,
  `evaluation/seed_voken/tokenize_to_jsonl_laion_ibq.py`,
  `evaluation/unitok/tokenize_to_jsonl_laion_unitok.py` (and the `journeydb`
  counterparts). BLIP3o is tokenized directly from its tar shards by the
  per-tokenizer `blip3o_<tok>.py` scripts. Weights are supplied per
  [evaluation/TOKENIZERS.md](evaluation/TOKENIZERS.md).

```bash
# LAION is tokenized shard-by-shard (part_1..12) and once for the val split.
python tokenize_to_jsonl_laion.py \
  --folder_path $DATA_ROOT/filtered_laion_aesthetics_parquet_recaptioned/part_8.parquet \
  --output_dir  $DATA_ROOT/laion_part_8_tokenized_$TOKENIZER_NAME
python tokenize_to_jsonl_laion.py \
  --folder_path $DATA_ROOT/filtered_laion_aesthetics_parquet_recaptioned/val.parquet \
  --output_dir  $DATA_ROOT/laion_val_tokenized_$TOKENIZER_NAME

# JourneyDB (train / validation) uses tokenize_to_jsonl_journeydb[_<tok>].py.

# BLIP3o is tokenized directly from tar shards by the tokenizer's blip3o_<tok>.py
# (run from that tokenizer's dir; GigaTok shown, others analogous):
#   BLIP3o-short (large) -> pretraining ;  BLIP3o-60k (small, ~60k) -> SFT.
cd $REPO_ROOT/evaluation/GigaTok
python blip3o_gigatok.py --input_pairs $DATA_ROOT/BLIP3o_60k_short \
  --source_name blip3o-60k-short --temp_path /tmp/blip3o_short_jsonl \
  --save_path $DATA_ROOT/blip3o_60k_short_tokenized_$TOKENIZER_NAME     # -> pretrain
python blip3o_gigatok.py --input_pairs $DATA_ROOT/BLIP3o_60k \
  --source_name blip3o-60k --temp_path /tmp/blip3o_60k_jsonl \
  --save_path $DATA_ROOT/blip3o_60k_tokenized_$TOKENIZER_NAME           # -> SFT
```

(The `60k-short` / `blip3o-60k-short` names above are the scripts' internal
labels for **BLIP3o-short**.)

### 3b. Assemble → HF datasets

Flatten the per-rank JSONL into one folder, pack it into an Arrow dataset, split
off train/val (and an SFT subset), and convert pretrain records to SFT format:

```bash
# LAION-Aesthetics
mkdir $DATA_ROOT/laion_tokenized_$TOKENIZER_NAME
mv $DATA_ROOT/laion_part_{1..12}_tokenized_$TOKENIZER_NAME $DATA_ROOT/laion_tokenized_$TOKENIZER_NAME/
python laion_movejsonl.py --folder_path $DATA_ROOT/laion_tokenized_$TOKENIZER_NAME
python jsonl_to_arrow.py  --folder_path $DATA_ROOT/laion_tokenized_$TOKENIZER_NAME \
                          --save_path   $DATA_ROOT/laion_tokenized_${TOKENIZER_NAME}_hf --num_shards 128
```

- `*_movejsonl.py` — flatten `rank_*/batch_*.jsonl` into one directory.
- `jsonl_to_arrow.py` — pack JSONL into a sharded HuggingFace Arrow dataset.
- `tokenized_split.py` — split off a validation / SFT subset (`--val_sample N`).
- `pt_to_sft.py` — convert pretrain-format records into the SFT (instruction) format.

The **complete** per-dataset command sequences (LAION, JourneyDB, BLIP3o-short,
BLIP3o-60k, including the exact `--val_sample` counts) are in
[data_process/README.md](data_process/README.md).

This yields, per tokenizer, the datasets used in training, e.g.
`dclm30m_hf`, `laion_tokenized_<tok>_hf`, `journeydb_tokenized_<tok>_hf`,
`blip3o_short_tokenized_<tok>_hf`, and their `_sft_` counterparts.

## 4. SFT data

Supervised-finetuning image-text pairs (AI2D, DVQA, Mini-Gemini instructions,
LLaVA-Pretrain, ALLaVA captions, ...) are center-cropped and tokenized with
`convert_imagepair_cc512_sft.py`:

```bash
python convert_imagepair_cc512_sft.py \
  --input_pairs /path/to/data/playground/ai2d \
  --temp_path   /tmp/ai2d_jsonl \
  --save_path   $DATA_ROOT/ai2d_tokenized_$TOKENIZER_NAME \
  --vqgan_path  $REPO_ROOT/data/tokenizer \
  --source_name ai2d
```

The LAION / JourneyDB SFT portions are produced by the `tokenized_split.py` +
`pt_to_sft.py` steps in stage 3b. Pure-text instruction data (e.g. LMSYS-Chat)
is converted the same way as stage 1.

See [TRAIN.md](TRAIN.md) for how the resulting datasets are combined into the
continual-pretraining and SFT mixtures.
