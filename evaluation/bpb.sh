#!/bin/bash
# Bits-per-byte (validation loss) evaluation.
#
# Runs evaluation/bpb.py over one or more checkpoints of a trained model and
# reports task-specific validation losses (text / image / T2I / I2T). The
# tokenizer is inferred from the model name (see bpb.py / decode.py).
#
# --- path configuration (see .env.example) ---
set -a
[ -f ../.env ] && . ../.env
[ -f .env ] && . ./.env
set +a
: "${CKPT_ROOT:=/path/to/checkpoints}"
: "${DATA_ROOT:=/path/to/data}"
# ---------------------------------------------
cd "$(dirname "$0")"

NPROC="${NPROC:-8}"
# Name of the training run under $CKPT_ROOT (the tokenizer is parsed from it).
MODEL_NAME="${MODEL_NAME:-gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512}"
# One or more checkpoint sub-directories to evaluate.
CKPTS=(${CKPTS:-checkpoint-93752})

for ckpt in "${CKPTS[@]}"; do
    echo "=== bpb: ${MODEL_NAME} / ${ckpt} ==="
    torchrun --nproc_per_node="${NPROC}" bpb.py --ckpt "${ckpt}" --model_name "${MODEL_NAME}"
done
