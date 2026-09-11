#!/bin/bash
# --- path configuration (see .env.example) ---
set -a
[ -f .env ] && . ./.env
set +a
: "${REPO_ROOT:=/path/to/Tokenizer_UMM}"
: "${DATA_ROOT:=/path/to/data}"
: "${CKPT_ROOT:=/path/to/checkpoints}"
: "${HF_CACHE:=/path/to/hf_cache}"
: "${CONDA_ROOT:=/path/to/miniconda3}"
# ---------------------------------------------

CKPT=${CKPT:?"Error: CKPT environment variable is required"}

echo "Using checkpoint: $CKPT"

CUDA_VISIBLE_DEVICES=0 python3 -m model_vqa_loader \
    --model-path ${CKPT_ROOT}/$CKPT \
    --question-file ${DATA_ROOT}/playground/MME/llava_mme.jsonl \
    --image-folder ${DATA_ROOT}/playground/MME/MME_Benchmark_release_version \
    --answers-file ${DATA_ROOT}/playground/MME/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode chatml_direct

cd ${DATA_ROOT}/playground/MME

python convert_answer_to_mme.py --experiment ${CKPT}

cd eval_tool

python calculation.py --results_dir answers/${CKPT} > ${REPO_ROOT}/data/results/mme_${CKPT}.txt