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

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
 
CKPT=${CKPT:?"Error: CKPT environment variable is required"}

echo "Using checkpoint: $CKPT"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m model_vqa_loader \
        --model-path  ${CKPT_ROOT}/$CKPT \
        --question-file ${DATA_ROOT}/playground/pope/llava_pope_test.jsonl \
        --image-folder ${DATA_ROOT}/playground/pope/val2014  \
        --answers-file ${DATA_ROOT}/playground/pope/answers/${CKPT}_${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode chatml_direct &
done

wait

output_file=${DATA_ROOT}/playground/pope/answers/${CKPT}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${DATA_ROOT}/playground/pope/answers/${CKPT}_${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Remove the separate chunked answer files
for IDX in $(seq 0 $((CHUNKS-1))); do
    rm -f ${DATA_ROOT}/playground/pope/answers/${CKPT}_${CHUNKS}_${IDX}.jsonl
done

wait

python eval_pope.py \
    --annotation-dir ${DATA_ROOT}/playground/pope/coco \
    --question-file ${DATA_ROOT}/playground/pope/llava_pope_test.jsonl \
    --result-file $output_file \
    > ${REPO_ROOT}/data/results/pope_${CKPT}.txt
    
echo $CKPT