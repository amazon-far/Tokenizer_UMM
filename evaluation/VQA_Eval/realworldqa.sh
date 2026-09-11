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
 
REALWORLDQADIR="${DATA_ROOT}/playground/realworldqa"

CKPT=${CKPT:?"Error: CKPT environment variable is required"}

echo "Using checkpoint: $CKPT"


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m model_vqa_loader \
        --model-path  ${CKPT_ROOT}/$CKPT \
        --question-file $REALWORLDQADIR/questions.jsonl \
        --image-folder $REALWORLDQADIR/images \
        --answers-file $REALWORLDQADIR/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode chatml_direct &
done

wait
output_file=$REALWORLDQADIR/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $REALWORLDQADIR/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    rm -f $REALWORLDQADIR/answers/$CKPT/${CHUNKS}_${IDX}.jsonl
done

python eval_realworldqa.py \
    --annotation-file $REALWORLDQADIR/questions.jsonl \
    --result-file $REALWORLDQADIR/answers/$CKPT/merge.jsonl \
    --output-file $REALWORLDQADIR/outputs/$CKPT.jsonl \
    --csv-file $REALWORLDQADIR/outputs/$CKPT.csv \
    > ${REPO_ROOT}/data/results/realworldqa_${CKPT}.txt

echo $CKPT