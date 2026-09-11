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
SPLIT="llava_vqav2_mscoco_test-dev2015"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=${CKPT:?"Error: CKPT environment variable is required"}

echo "Using checkpoint: $CKPT"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m model_vqa_loader \
    --model-path  ${CKPT_ROOT}/$CKPT \
    --question-file ${DATA_ROOT}/playground/vqav2/$SPLIT.jsonl \
    --image-folder ${DATA_ROOT}/playground/vqav2/test2015 \
    --answers-file ${DATA_ROOT}/playground/vqav2/answers/${SPLIT}/${CKPT}/${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode chatml_direct &
done

wait

output_file=${DATA_ROOT}/playground/vqav2/answers/${SPLIT}/${CKPT}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${DATA_ROOT}/playground/vqav2/answers/${SPLIT}/${CKPT}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

wait

# Remove the separate chunked answer files
for IDX in $(seq 0 $((CHUNKS-1))); do
    rm -f ${DATA_ROOT}/playground/vqav2/answers/${SPLIT}/${CKPT}/${CHUNKS}_${IDX}.jsonl
done

python convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT --dir ${DATA_ROOT}/playground/vqav2/

cp ${DATA_ROOT}/playground/vqav2/answers_upload/${SPLIT}/${CKPT}.json ${REPO_ROOT}/data/results/vqav2_${CKPT}.json

echo $CKPT