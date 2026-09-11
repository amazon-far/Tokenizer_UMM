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
        --question-file ${DATA_ROOT}/playground/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder ${DATA_ROOT}/playground/textvqa/train_images \
        --answers-file ${DATA_ROOT}/playground/textvqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode chatml_direct &
done

wait

output_file=${DATA_ROOT}/playground/textvqa/answers/${CKPT}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${DATA_ROOT}/playground/textvqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    rm -f ${DATA_ROOT}/playground/textvqa/answers/$CKPT/${CHUNKS}_${IDX}.jsonl
done

python -m  eval_textvqa \
    --annotation-file ${DATA_ROOT}/playground/textvqa/TextVQA_0.5.1_val.json \
    --result-file $output_file \
    > ${REPO_ROOT}/data/results/textvqa_${CKPT}.txt
echo $CKPT