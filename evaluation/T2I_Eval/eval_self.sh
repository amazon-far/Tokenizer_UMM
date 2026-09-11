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

CUDA_VISIBLE_DEVICES='0'

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="${CKPT_ROOT}/gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_1024/acheckpoint-46880"
SAVE_PTH="${REPO_ROOT}/data/images/"
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python self_generation.py \
        --model_path $CKPT \
        --save_path $SAVE_PTH \
        --image_length "${IMAGE_LENGTH:-256}" \
        --max_image_token_id "${MAX_IMAGE_TOKEN_ID:-16383}" \
        --cfg_scale 7.0 \
        --tau 1.0 \
        --topk 0 \
        --topp 1.0 \
        --batch_size 128 \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
done
wait
source ${CONDA_ROOT}/etc/profile.d/conda.sh
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} conda run -n liquid python decode.py \
        --model_path $CKPT \
        --save_path $SAVE_PTH \
        --benchmark_name genai \
        --chunk_idx $IDX &
done
wait