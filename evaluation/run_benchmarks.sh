#!/bin/bash
# Run the full downstream benchmark suite for a trained checkpoint:
#   - Text-to-image generation + scoring: GenAI-Bench, MJHQ-30K
#   - Visual understanding (VQA): VQAv2 / GQA / TextVQA / POPE
#   - Validation loss: bits-per-byte
#
# Set the checkpoint(s) inside each sub-script (or via CKPT / CKPTNAMES) and the
# tokenizer weights per evaluation/TOKENIZERS.md before running.
#
# --- path configuration (see .env.example) ---
set -a
[ -f ../.env ] && . ../.env
[ -f .env ] && . ./.env
set +a
: "${REPO_ROOT:=/path/to/Tokenizer_UMM}"
: "${CKPT_ROOT:=/path/to/checkpoints}"
# ---------------------------------------------
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

# 1) Text-to-image generation + metrics
cd "${HERE}/T2I_Eval"
bash eval_mjhq.sh          # MJHQ-30K (gFID)
bash eval_genai.sh         # GenAI-Bench (VQAScore)

# 2) Visual understanding (VQA) benchmarks
cd "${HERE}/VQA_Eval"
bash qa.sh                 # or: CKPT="<exp_name>" sh testall.sh

# 3) Validation loss (bits-per-byte)
cd "${HERE}"
bash bpb.sh
