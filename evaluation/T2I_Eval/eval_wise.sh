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

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
# Define your checkpoints as an array
CHECKPOINTS=(
    "${CKPT_ROOT}/qwen3_8b_mixpretrain_stage2_1_2_0.8_1e-4_0.03_blip3o_lmsys_ptimg_blip3o_test_ep2"
)

# Corresponding checkpoint names (optional - can auto-extract from path)
CKPTNAMES=(
    "qwen3_8b_mixpretrain_stage2_1_2_0.8_1e-4_0.03_blip3o_lmsys_ptimg_blip3o_test_ep2"
) 

# Loop through each checkpoint
for i in "${!CHECKPOINTS[@]}"; do
    CKPT="${CHECKPOINTS[$i]}"
    CKPTNAME="${CKPTNAMES[$i]}"
    SAVE_PTH="${DATA_ROOT}/playground/WISE-cfg_7.0-topk_0-topp_1.0-tau_1.0_${CKPTNAME}"
    rm -rf $SAVE_PTH
    mkdir -p ${SAVE_PTH}


    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python wise_generation.py \
            --model_path $CKPT \
            --save_path $SAVE_PTH \
            --image_length "${IMAGE_LENGTH:-256}" \
            --max_image_token_id "${MAX_IMAGE_TOKEN_ID:-16383}" \
            --cfg_scale 7.0 \
            --tau 1.0 \
            --topk 0 \
            --topp 1.0 \
            --batch_size 64 \
            --num_chunks $CHUNKS \
            --chunk_idx $IDX &
    done
    wait
    rm -rf ${SAVE_PTH}/Results
    source ${CONDA_ROOT}/etc/profile.d/conda.sh
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} conda run -n liquid python decode.py \
            --model_path $CKPT \
            --save_path $SAVE_PTH \
            --benchmark_name wise \
            --chunk_idx $IDX &
    done
    wait
    conda run -n base python eval_wise.py \
        --json_path WISE/cultural_common_sense.json \
        --output_dir ${SAVE_PTH}/Results/cultural_common_sense \
        --image_dir ${SAVE_PTH} \
        --api_key "${OPENAI_API_KEY}" \
        --model "gpt-4o-2024-05-13" \
        --result_full ${SAVE_PTH}/Results/cultural_common_sense_full_results.json \
        --result_scores ${SAVE_PTH}/Results/cultural_common_sense_scores_results.jsonl \
        --max_workers 96

    conda run -n base python eval_wise.py \
        --json_path WISE/spatio-temporal_reasoning.json \
        --output_dir ${SAVE_PTH}/Results/spatio-temporal_reasoning \
        --image_dir ${SAVE_PTH} \
        --api_key "${OPENAI_API_KEY}" \
        --model "gpt-4o-2024-05-13" \
        --result_full ${SAVE_PTH}/Results/spatio-temporal_reasoning_results.json \
        --result_scores ${SAVE_PTH}/Results/spatio-temporal_reasoning_results.jsonl \
        --max_workers 96

    conda run -n base python eval_wise.py \
        --json_path WISE/natural_science.json \
        --output_dir ${SAVE_PTH}/Results/natural_science \
        --image_dir ${SAVE_PTH} \
        --api_key "${OPENAI_API_KEY}" \
        --model "gpt-4o-2024-05-13" \
        --result_full ${SAVE_PTH}/Results/natural_science_full_results.json \
        --result_scores ${SAVE_PTH}/Results/natural_science_scores_results.jsonl \
        --max_workers 96

    python calculate_wise.py \
        "${SAVE_PTH}/Results/cultural_common_sense_scores_results.jsonl" \
        "${SAVE_PTH}/Results/natural_science_scores_results.jsonl" \
        "${SAVE_PTH}/Results/spatio-temporal_reasoning_results.jsonl" \
        --category all \
        > ${REPO_ROOT}/data/results/wise_$CKPTNAME.txt
    echo "Completed checkpoint: $CKPTNAME"
    echo ""
done

echo "All checkpoints processed!"