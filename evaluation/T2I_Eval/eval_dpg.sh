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



# Evaluation phase
PIC_NUM=${PIC_NUM:-4}
PROCESSES=${PROCESSES:-8}
PORT=${PORT:-29500}


# Loop through each checkpoint
for i in "${!CHECKPOINTS[@]}"; do
    CKPT="${CHECKPOINTS[$i]}"
    CKPTNAME="${CKPTNAMES[$i]}"
    SAVE_PTH="${DATA_ROOT}/playground/DPG-cfg_7.0-topk_0-topp_1.0-tau_1.0_${CKPTNAME}"
    
    echo "=========================================="
    echo "Processing checkpoint $((i+1))/${#CHECKPOINTS[@]}: $CKPTNAME"
    echo "=========================================="
    
    rm -rf $SAVE_PTH
    mkdir -p ${SAVE_PTH}

    # Generation phase
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python dpgbench_generation.py \
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
    source ${CONDA_ROOT}/etc/profile.d/conda.sh
    # Decode phase
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} conda run -n liquid python decode.py \
            --model_path $CKPT \
            --save_path $SAVE_PTH \
            --benchmark_name dpg \
            --chunk_idx $IDX &
    done
    wait
    conda run -n dpg accelerate launch --num_machines 1 --num_processes $PROCESSES --multi_gpu --mixed_precision "fp16" --main_process_port $PORT compute_dpg_bench.py \
        --image-root-path $SAVE_PTH \
        --resolution 256 \
        --pic-num $PIC_NUM \
        --vqa-model mplug \
        > ${REPO_ROOT}/data/results/dpg_$CKPTNAME.txt
    
    echo "Completed checkpoint: $CKPTNAME"
    echo ""
done

echo "All checkpoints processed!"