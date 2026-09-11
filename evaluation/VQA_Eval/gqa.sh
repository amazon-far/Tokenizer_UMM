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
 
SPLIT="llava_gqa_testdev_balanced"
GQADIR="${DATA_ROOT}/playground/gqa"

CKPT=${CKPT:?"Error: CKPT environment variable is required"}

echo "Using checkpoint: $CKPT"
 

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m model_vqa_loader \
        --model-path  ${CKPT_ROOT}/$CKPT \
        --question-file $GQADIR/$SPLIT.jsonl \
        --image-folder $GQADIR/images \
        --answers-file $GQADIR/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode chatml_direct &
done

wait

 


output_file=$GQADIR/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $GQADIR/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    rm -f $GQADIR/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl
done

python convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval.py --tier testdev_balanced > ${REPO_ROOT}/data/results/gqa_${CKPT}.txt
echo $CKPT