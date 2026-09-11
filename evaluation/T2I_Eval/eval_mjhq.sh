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
    # "${CKPT_ROOT}/ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/lcheckpoint-35157"
    # "${CKPT_ROOT}/ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/lcheckpoint-46876"
    # "${CKPT_ROOT}/ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/lcheckpoint-58595"
    # "${CKPT_ROOT}/ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/lcheckpoint-70314"
    # "${CKPT_ROOT}/ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/lcheckpoint-82033"
    # "${CKPT_ROOT}/ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/lcheckpoint-23438"
    # "${CKPT_ROOT}/ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/lcheckpoint-35157"
    # "${CKPT_ROOT}/ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/lcheckpoint-46876"
    # "${CKPT_ROOT}/ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/lcheckpoint-58595"
    # "${CKPT_ROOT}/ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/lcheckpoint-70314"
    # "${CKPT_ROOT}/ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/lcheckpoint-82033"
    # "${CKPT_ROOT}/ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/checkpoint-93752"
    # "${CKPT_ROOT}/ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/lcheckpoint-5860"
    # "${CKPT_ROOT}/ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/lcheckpoint-8790"
    # "${CKPT_ROOT}/ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/lcheckpoint-11720"
    # "${CKPT_ROOT}/ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/lcheckpoint-14650"
    # "${CKPT_ROOT}/ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/lcheckpoint-17580"
    # "${CKPT_ROOT}/ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/lcheckpoint-20510"
    # "${CKPT_ROOT}/ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/lcheckpoint-35157"
    # "${CKPT_ROOT}/ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/lcheckpoint-46876"
    # "${CKPT_ROOT}/ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/lcheckpoint-58595"
    # "${CKPT_ROOT}/ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/lcheckpoint-70314"
    # "${CKPT_ROOT}/ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/lcheckpoint-82033"
    # "${CKPT_ROOT}/ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/checkpoint-23438"
    # "${CKPT_ROOT}/ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/lcheckpoint-35157"
    # "${CKPT_ROOT}/ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/lcheckpoint-46876"
    # "${CKPT_ROOT}/ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/lcheckpoint-58595"
    # "${CKPT_ROOT}/ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/lcheckpoint-70314"
    # "${CKPT_ROOT}/ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/lcheckpoint-82033"
    # "${CKPT_ROOT}/ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/lcheckpoint-93752"
    # "${CKPT_ROOT}/ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/checkpoint-5860"
    # "${CKPT_ROOT}/ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/lcheckpoint-8790"
    # "${CKPT_ROOT}/ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/lcheckpoint-11720"
    # "${CKPT_ROOT}/ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/lcheckpoint-14650"
    # "${CKPT_ROOT}/ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/lcheckpoint-17580"
    # "${CKPT_ROOT}/ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/lcheckpoint-20510"
    # "${CKPT_ROOT}/ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/lcheckpoint-23438"
    # "${CKPT_ROOT}/ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/lcheckpoint-35157"
    # "${CKPT_ROOT}/ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/lcheckpoint-46876"
    # "${CKPT_ROOT}/ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/lcheckpoint-58595"
    # "${CKPT_ROOT}/ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/lcheckpoint-70314"
    # "${CKPT_ROOT}/ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/lcheckpoint-82033"
    # "${CKPT_ROOT}/ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/lcheckpoint-23438"
    # "${CKPT_ROOT}/ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/checkpoint-35157"
    # "${CKPT_ROOT}/ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/lcheckpoint-46876"
    # "${CKPT_ROOT}/ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/lcheckpoint-58595"
    # "${CKPT_ROOT}/ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/lcheckpoint-70314"
    # "${CKPT_ROOT}/ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/lcheckpoint-82033"
    # "${CKPT_ROOT}/ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512/lcheckpoint-93752"
    # "${CKPT_ROOT}/ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/checkpoint-5860"
    # "${CKPT_ROOT}/ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/lcheckpoint-8790"
    # "${CKPT_ROOT}/ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/lcheckpoint-11720"
    # "${CKPT_ROOT}/ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/lcheckpoint-14650"
    # "${CKPT_ROOT}/ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/lcheckpoint-17580"
    # "${CKPT_ROOT}/ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/lcheckpoint-20510"
    # "${CKPT_ROOT}/gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/lcheckpoint-5860"
    # "${CKPT_ROOT}/gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/lcheckpoint-8790"
    # "${CKPT_ROOT}/gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/lcheckpoint-11720"
    # "${CKPT_ROOT}/gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/lcheckpoint-14650"
    # "${CKPT_ROOT}/gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/lcheckpoint-17580"
    # "${CKPT_ROOT}/gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/lcheckpoint-20510"
    "${CKPT_ROOT}/unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/checkpoint-23418"
    "${CKPT_ROOT}/unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/checkpoint-35127"
    "${CKPT_ROOT}/unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/checkpoint-46836"
    "${CKPT_ROOT}/unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/checkpoint-58545"
    "${CKPT_ROOT}/unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/checkpoint-70254"
    "${CKPT_ROOT}/unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/checkpoint-81963"
    "${CKPT_ROOT}/unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/checkpoint-93672"
    "${CKPT_ROOT}/unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/checkpoint-5856"
    "${CKPT_ROOT}/unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/checkpoint-8784"
    "${CKPT_ROOT}/unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/checkpoint-11712"
    "${CKPT_ROOT}/unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/checkpoint-14640"
    "${CKPT_ROOT}/unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/checkpoint-17568"
    "${CKPT_ROOT}/unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/checkpoint-20496"
    "${CKPT_ROOT}/unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/checkpoint-23424"
    "${CKPT_ROOT}/unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/checkpoint-23418"
    "${CKPT_ROOT}/unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/checkpoint-35127"
    "${CKPT_ROOT}/unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/checkpoint-46836"
    "${CKPT_ROOT}/unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/checkpoint-58545"
    "${CKPT_ROOT}/unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/checkpoint-70254"
    "${CKPT_ROOT}/unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/checkpoint-81963"
    "${CKPT_ROOT}/unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512/checkpoint-93672"
    "${CKPT_ROOT}/unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/checkpoint-5856"
    "${CKPT_ROOT}/unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/checkpoint-8784"
    "${CKPT_ROOT}/unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/checkpoint-11712"
    "${CKPT_ROOT}/unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/checkpoint-14640"
    "${CKPT_ROOT}/unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/checkpoint-17568"
    "${CKPT_ROOT}/unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/checkpoint-20496"
    "${CKPT_ROOT}/unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048/checkpoint-23424"
)

# Corresponding checkpoint names (optional - can auto-extract from path)
CKPTNAMES=(
    # "ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_35157"
    # "ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_46876"
    # "ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_58595"
    # "ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_70314"
    # "ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_82033"
    # "ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_23438"
    # "ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_35157"
    # "ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_46876"
    # "ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_58595"
    # "ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_70314"
    # "ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_82033"
    # "ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_93752"
    # "ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_5860"
    # "ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_8790"
    # "ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_11720"
    # "ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_14650"
    # "ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_17580"
    # "ibq_1024_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_20510"
    # "ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_35157"
    # "ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_46876"
    # "ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_58595"
    # "ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_70314"
    # "ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_82033"
    # "ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_23438"
    # "ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_35157"
    # "ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_46876"
    # "ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_58595"
    # "ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_70314"
    # "ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_82033"
    # "ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_93752"
    # "ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_5860"
    # "ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_8790"
    # "ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_11720"
    # "ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_14650"
    # "ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_17580"
    # "ibq_8192_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_20510"
    # "ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_23438"
    # "ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_35157"
    # "ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_46876"
    # "ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_58595"
    # "ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_70314"
    # "ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_82033"
    # "ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_23438"
    # "ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_35157"
    # "ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_46876"
    # "ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_58595"
    # "ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_70314"
    # "ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_82033"
    # "ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_512_93752"
    # "ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_5860"
    # "ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_8790"
    # "ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_11720"
    # "ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_14650"
    # "ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_17580"
    # "ibq_16384_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_20510"
    # "gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_5860"
    # "gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_8790"
    # "gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_11720"
    # "gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_14650"
    # "gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_17580"
    # "gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_20510"
    "unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_23418"
    "unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_35127"
    "unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_46836"
    "unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_58545"
    "unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_70254"
    "unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_81963"
    "unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_93672"
    "unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_5856"
    "unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_8784"
    "unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_11712"
    "unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_14640"
    "unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_17568"
    "unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_20496"
    "unitok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_23424"
    "unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_23418"
    "unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_35127"
    "unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_46836"
    "unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_58545"
    "unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_70254"
    "unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_81963"
    "unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_512_93672"
    "unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_5856"
    "unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_8784"
    "unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_11712"
    "unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_14640"
    "unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_17568"
    "unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_20496"
    "unitok_sem_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_3e-5_2048_23424"
) 


# Loop through each checkpoint
for i in "${!CHECKPOINTS[@]}"; do
    CKPT="${CHECKPOINTS[$i]}"
    CKPTNAME="${CKPTNAMES[$i]}"
    SAVE_PTH="${DATA_ROOT}/playground/mjhq_results_${CKPTNAME}"
    echo "=========================================="
    echo "Processing checkpoint $((i+1))/${#CHECKPOINTS[@]}: $CKPTNAME"
    echo "=========================================="
    
    rm -rf $SAVE_PTH
    mkdir -p ${SAVE_PTH}

    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python mjhq_generation.py \
            --model_path $CKPT \
            --save_path $SAVE_PTH \
            --image_length "${IMAGE_LENGTH:-256}" \
            --max_image_token_id "${MAX_IMAGE_TOKEN_ID:-16383}" \
            --cfg_scale 7.0 \
            --batch_size 64 \
            --tau 1.0 \
            --topk 0 \
            --topp 1.0 \
            --num_chunks $CHUNKS \
            --chunk_idx $IDX &
    done
    wait

    source ${CONDA_ROOT}/etc/profile.d/conda.sh

    # Decode phase
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} conda run -n liquid python decode.py \
            --model_path $CKPT \
            --save_path $SAVE_PTH/CFG7.0_topk0_topp1.0_tau_1.0 \
            --benchmark_name mjhq \
            --chunk_idx $IDX &
    done
    wait

    # Evaluation phase
    python get_fid.py \
        --src_dir ${DATA_ROOT}/playground/MJHQ-30K/mjhq30k_imgs \
        --gen_dir $SAVE_PTH/CFG7.0_topk0_topp1.0_tau_1.0/ \
        > ${REPO_ROOT}/data/results/mjhq_${CKPTNAME}.txt
    
    echo "Completed checkpoint: $CKPTNAME"
    echo ""
    rm -rf $SAVE_PTH
done

echo "All checkpoints processed!"
