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

# Define your checkpoints as an array
CHECKPOINTS=(
    # "${CKPT_ROOT}/gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_1024/lcheckpoint-11720"
    # "${CKPT_ROOT}/gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_1024/lcheckpoint-23440"
    # "${CKPT_ROOT}/gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_1024/lcheckpoint-46880"
    # "${CKPT_ROOT}/gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_1024/lcheckpoint-11720"
    # "${CKPT_ROOT}/gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_1024/checkpoint-23440"
    # "${CKPT_ROOT}/gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/lcheckpoint-23438"
    # "${CKPT_ROOT}/gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/lcheckpoint-46876"
    # "${CKPT_ROOT}/gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/checkpoint-93752"
    # "${CKPT_ROOT}/gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/lcheckpoint-23438"
    # "${CKPT_ROOT}/gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/lcheckpoint-46876"
    # "${CKPT_ROOT}/gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/checkpoint-93752"
    # "${CKPT_ROOT}/gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_2048/lcheckpoint-5860"
    # "${CKPT_ROOT}/gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_2048/lcheckpoint-11720"
    # "${CKPT_ROOT}/gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_2048/checkpoint-23440"
    # "${CKPT_ROOT}/gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/lcheckpoint-5860"
    # "${CKPT_ROOT}/gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/checkpoint-23440"
    # "${CKPT_ROOT}/qwen3_0.6b_mixpretrain_stage2_1_2_0.8_1e-4_0.03_blip3o_lmsys_ptimg_blip3o_test_ep2"
    # "${CKPT_ROOT}/qwen3_8b_mixpretrain_stage2_1_1_0.8_1e-4_0.03_blip3o_lmsys_ptimg_blip3o_test_ep2"
)

# Corresponding checkpoint names (optional - can auto-extract from path)
CKPTNAMES=(
    # "gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_1024_11720"
    # "gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_1024_23440"
    # "gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_1024_46880"
    # "gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_1024_11720"
    # "gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_1024_23440"
    # "gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_23438"
    # "gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_46876"
    # "gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_93752"
    # "gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_23438"
    # "gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_46876"
    # "gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_93752"
    # "gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_2048_5860"
    # "gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_2048_11720"
    # "gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_2048_23440"
    # "gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_5860"
    # "gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_23440"
    # "qwen3_0.6b_mixpretrain_stage2_1_2_0.8_1e-4_0.03_blip3o_lmsys_ptimg_blip3o_test_ep2"
    # "qwen3_8b_mixpretrain_stage2_1_1_0.8_1e-4_0.03_blip3o_lmsys_ptimg_blip3o_test_ep2"
) 

for i in "${!CHECKPOINTS[@]}"; do
    CKPT="${CHECKPOINTS[$i]}"
    CKPTNAME="${CKPTNAMES[$i]}"

    accelerate launch  --main_process_port 9999 -m lm_eval  --model hf \
        --model_args pretrained=${CKPT},dtype="float" \
        --tasks  hellaswag,winogrande,arc_easy,arc_challenge,boolq,mmlu \
        --batch_size 8 \
        --output_path ${REPO_ROOT}/data/results/ \
        > ${REPO_ROOT}/data/results/qa_${CKPTNAME}.txt
    wait
done

# --num_fewshot 5 \