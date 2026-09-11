# Evaluation

The unified model can be evaluated on text-to-image generation, visual
understanding (VQA), and language tasks. Because the model is an extension of a
standard HuggingFace LLM, evaluation runs on the `transformers` library without
special dependencies.

Evaluation runs in the same `liquid` environment as training, built by
[`scripts/setup_env.sh`](../scripts/setup_env.sh):

```bash
python=3.10
torch=2.1.0
flash-attention=2.5.8
transformers=4.51.0
```

`transformers>=4.51.0` is **required**, not just recommended: Qwen3 support
first shipped in 4.51.0, so earlier releases cannot load these checkpoints at
all. Newer releases generally work.

## Preparation

- **Checkpoint.** Train a model following [TRAIN.md](../TRAIN.md), or point the
  scripts at your own checkpoint. The eval scripts refer to checkpoints via
  `CKPT_ROOT` (see [README](../README.md#path-configuration)); paths below use
  `/path/to/checkpoint` as a placeholder.
- **Tokenizer weights.** Download the weights for whichever image tokenizer the
  checkpoint was trained with and place them at the expected paths — see
  [TOKENIZERS.md](TOKENIZERS.md).

The tokenizer is selected automatically from the checkpoint path (the decoding
and loading code branches on substrings such as `gigatok`, `ibq`, `unitok`;
otherwise it falls back to the Chameleon VQGAN).

### Generation settings

Each T2I script edits a `CHECKPOINTS` / `CKPTNAMES` array at the top — point
these at your own runs before launching. The number of image tokens per image
and the codebook bound are shared knobs, defaulting to the 256-token /
16384-codebook setting used throughout the paper:

```bash
# defaults: 16x16 = 256 tokens, codebook 16384 (GigaTok, GigaTok-DINO, IBQ-16384)
IMAGE_LENGTH=256 MAX_IMAGE_TOKEN_ID=16383 bash eval_genai.sh

# for a smaller codebook, e.g. IBQ-8192 / IBQ-1024
MAX_IMAGE_TOKEN_ID=8191 bash eval_genai.sh
```

`MAX_IMAGE_TOKEN_ID` must be `codebook_size - 1` for the tokenizer the
checkpoint was trained with, and `IMAGE_LENGTH` must match the token grid
(256 for 16×16 at 256px). A mismatch produces garbage images rather than an
error.

## Text-to-image evaluation

### GenAI-Bench

The GenAI-Bench-527 prompt files (`prompts.txt`, `genai_skills.json`) are
already committed under `T2I_Eval/`; re-download them from
[zhiqiulin/GenAI-Bench-527](https://huggingface.co/datasets/zhiqiulin/GenAI-Bench-527)
only if you want a newer revision.

```bash
cd T2I_Eval

# generate images
bash eval_genai.sh

# score on GenAI-Bench (run in the `t2v` env, which is built by scripts/setup_env.sh)
conda activate t2v
python eval_genai_527.py --image_dir $SAVE_PTH
```

### MJHQ-30K

```bash
cd T2I_Eval

huggingface-cli download --repo-type dataset --resume-download playgroundai/MJHQ-30K --local-dir MJHQ-30K
cd MJHQ-30K && unzip -q mjhq30k_imgs.zip -d mjhq30k_imgs && cd ..

pip install clean-fid scipy==1.11.1

# Generating 30k images is time-consuming; use a large batch size and/or
# distribute across machines.
bash eval_mjhq.sh
```

### DPG-Bench

[`eval_dpg.sh`](T2I_Eval/eval_dpg.sh) generates `PIC_NUM` images per prompt,
decodes them, and scores with `compute_dpg_bench.py` (mPLUG VQA).

```bash
cd T2I_Eval
PIC_NUM=4 bash eval_dpg.sh
```

> Scoring runs in a separate `dpg` conda environment (`conda run -n dpg`) that
> `scripts/setup_env.sh` does **not** build — the mPLUG scorer has dependencies
> that conflict with the `liquid` env. Create it yourself following
> [DPG-Bench](https://github.com/TencentQQGYLab/ELLA#-dpg-bench) before running
> the scoring step. Results are written to `$REPO_ROOT/data/results/dpg_*.txt`.

### WISE

[`eval_wise.sh`](T2I_Eval/eval_wise.sh) generates and decodes images, scores the
three WISE subsets with a GPT judge, then aggregates with `calculate_wise.py`.

```bash
cd T2I_Eval
export OPENAI_API_KEY=...        # scoring uses gpt-4o via the `base` env
bash eval_wise.sh
```

> Requires the WISE prompt files at `T2I_Eval/WISE/{cultural_common_sense,
> spatio-temporal_reasoning,natural_science}.json`, which are **not** shipped
> here — download them from [PKU-YuanGroup/WISE](https://github.com/PKU-YuanGroup/WISE).
> Scoring calls the OpenAI API and therefore costs money; the `base` env with
> `openai==0.28.0` is built by `scripts/setup_env.sh`. Results are written to
> `$REPO_ROOT/data/results/wise_*.txt`.

### Self-generated samples

[`eval_self.sh`](T2I_Eval/eval_self.sh) generates images for the ad-hoc prompts
in `T2I_Eval/myprompts.txt` (single GPU) and decodes them into
`$REPO_ROOT/data/images/` — useful for eyeballing a checkpoint without running a
full benchmark.

```bash
cd T2I_Eval && bash eval_self.sh
```

## VQA benchmarks

Follow [LLaVA's evaluation guide](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)
for benchmark data preparation, then:

```bash
cd VQA_Eval
bash textvqa.sh
bash gqa.sh
bash pope.sh
bash vqav2.sh
```

## Validation loss (bits-per-byte)

Task-specific validation losses (text / image / T2I / I2T) — the main analysis
signal in the paper — are computed by `bpb.py`, wrapped by
[`bpb.sh`](bpb.sh):

```bash
cd evaluation
MODEL_NAME=<exp_name> CKPTS="checkpoint-93752 checkpoint-46876" bash bpb.sh
```

`MODEL_NAME` is the training-run directory under `$CKPT_ROOT` (the tokenizer is
parsed from it) and `CKPTS` is a space-separated list of checkpoint
sub-directories.

## Full benchmark suite

To run T2I generation + scoring, VQA, and bits-per-byte for one checkpoint in
sequence, use [`run_benchmarks.sh`](run_benchmarks.sh) (set the checkpoint and
tokenizer weights first):

```bash
bash evaluation/run_benchmarks.sh
```

## Language tasks

The model can be used directly as a language model; we report results with
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

`lm-eval` is installed in the `liquid` env by `scripts/setup_env.sh`.

```bash
lm_eval --model hf \
    --model_args pretrained=/path/to/checkpoint,dtype="float" \
    --tasks hellaswag,winogrande,arc_easy,arc_challenge,boolq,mmlu \
    --device cuda:0 --batch_size 8
```
