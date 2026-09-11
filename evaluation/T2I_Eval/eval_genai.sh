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

CHECKPOINTS=(
    # "${CKPT_ROOT}/gigatok_qwen3_1.7b_mixpretrain_stage1_1_8_0.8_60b_1e-5_512/checkpoint-23438"
    # "${CKPT_ROOT}/gigatok_qwen3_1.7b_mixpretrain_stage1_1_8_0.8_60b_1e-5_512/checkpoint-35157"
    # "${CKPT_ROOT}/gigatok_qwen3_4b_mixpretrain_stage1_1_8_0.8_60b_1e-5_512/checkpoint-23438"
    # "${CKPT_ROOT}/gigatok_qwen3_4b_mixpretrain_stage1_1_8_0.8_60b_1e-5_512/checkpoint-35157"
    # "${CKPT_ROOT}/gigatok_qwen3_1.7b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/checkpoint-23438"
    # "${CKPT_ROOT}/gigatok_qwen3_1.7b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/checkpoint-35157"
    # "${CKPT_ROOT}/gigatok_qwen3_4b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/checkpoint-23438"
    # "${CKPT_ROOT}/gigatok_qwen3_4b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/checkpoint-35157"
    # "${CKPT_ROOT}/gigatok_qwen3_1.7b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/checkpoint-23438"
    # "${CKPT_ROOT}/gigatok_qwen3_1.7b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/checkpoint-35157"
    # "${CKPT_ROOT}/gigatok_qwen3_4b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/checkpoint-23438"
    # "${CKPT_ROOT}/gigatok_qwen3_4b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/checkpoint-35157"
    # "${CKPT_ROOT}/gigatok_dino_qwen3_1.7b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/checkpoint-23438"
    # "${CKPT_ROOT}/gigatok_dino_qwen3_1.7b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/checkpoint-35157"
    # "${CKPT_ROOT}/gigatok_dino_qwen3_4b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/checkpoint-23438"
    # "${CKPT_ROOT}/gigatok_dino_qwen3_4b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/checkpoint-35157"
    
)

# Corresponding checkpoint names (optional - can auto-extract from path)
CKPTNAMES=(
    # "gigatok_qwen3_1.7b_mixpretrain_stage1_1_8_0.8_60b_1e-5_512_23438"
    # "gigatok_qwen3_1.7b_mixpretrain_stage1_1_8_0.8_60b_1e-5_512_35157"
    # "gigatok_qwen3_4b_mixpretrain_stage1_1_8_0.8_60b_1e-5_512_23438"
    # "gigatok_qwen3_4b_mixpretrain_stage1_1_8_0.8_60b_1e-5_512_35157"
    # "gigatok_qwen3_1.7b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_23438"
    # "gigatok_qwen3_1.7b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_35157"
    # "gigatok_qwen3_4b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_23438"
    # "gigatok_qwen3_4b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_35157"
    # "gigatok_qwen3_1.7b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_23438"
    # "gigatok_qwen3_1.7b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_35157"
    # "gigatok_qwen3_4b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_23438"
    # "gigatok_qwen3_4b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_35157"
    # "gigatok_dino_qwen3_1.7b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_23438"
    # "gigatok_dino_qwen3_1.7b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_35157"
    # "gigatok_dino_qwen3_4b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_23438"
    # "gigatok_dino_qwen3_4b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_35157"
) 


# Loop through each checkpoint
for i in "${!CHECKPOINTS[@]}"; do
    CKPT="${CHECKPOINTS[$i]}"
    CKPTNAME="${CKPTNAMES[$i]}"
    SAVE_PTH="${DATA_ROOT}/playground/GenAI-cfg_7.0-topk_0-topp_1.0-tau_1.0_${CKPTNAME}"
    
    echo "=========================================="
    echo "Processing checkpoint $((i+1))/${#CHECKPOINTS[@]}: $CKPTNAME"
    echo "=========================================="
    
    rm -rf $SAVE_PTH
    mkdir -p ${SAVE_PTH}

    # Generation phase
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python genaibench_generation.py \
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
    # Decode phase
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} conda run -n liquid python decode.py \
            --model_path $CKPT \
            --save_path $SAVE_PTH \
            --benchmark_name genai \
            --chunk_idx $IDX &
    done
    wait

    # Evaluation phase
    conda run -n t2v python eval_genai_527.py \
        --image_dir $SAVE_PTH/ \
        > ${REPO_ROOT}/data/results/genai_$CKPTNAME.txt
    
    echo "Completed checkpoint: $CKPTNAME"
    echo ""
done

echo "All checkpoints processed!"