#!/bin/bash
# Supervised fine-tuning (stage 2) on multimodal instruction / captioning data.
#
# Starts from a stage-1 continual-pretraining checkpoint and trains for 2 epochs
# with a cosine schedule. Data must be pre-tokenized with the same tokenizer as
# the base checkpoint.
#
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
export WANDB_PROJECT="${WANDB_PROJECT:-tokenizer_umm}"

# ---- experiment configuration (edit for your run) ----
SIZE="${SIZE:-0.6b}"
TOKENIZER="${TOKENIZER:-gigatok}"
BASE_CKPT="${BASE_CKPT:-${CKPT_ROOT}/${TOKENIZER}_qwen3_${SIZE}_mixpretrain/acheckpoint-23440}"  # stage-1 checkpoint
EXP_NAME="${EXP_NAME:-${TOKENIZER}_qwen3_${SIZE}_sft}"

# ---- distributed launch (defaults to a single 8-GPU node) ----
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-2935}"
NPROC="${NPROC:-8}"

# ---- global batch size ----
# SFT uses a global batch of 1024; gradient accumulation is derived from the
# number of GPUs available: BS x NPROC x NNODES x GRAD_ACCUM.
BS="${BS:-16}"                       # per-device batch size
GLOBAL_BS="${GLOBAL_BS:-1024}"
_per_step=$(( BS * NPROC * NNODES ))
if [ $(( GLOBAL_BS % _per_step )) -ne 0 ]; then
    echo "ERROR: GLOBAL_BS=${GLOBAL_BS} is not divisible by BS*NPROC*NNODES=${_per_step}." >&2
    echo "       Adjust BS, NPROC, NNODES or GLOBAL_BS so they divide evenly." >&2
    exit 1
fi
GRAD_ACCUM=$(( GLOBAL_BS / _per_step ))
echo "global batch ${GLOBAL_BS} = BS ${BS} x NPROC ${NPROC} x NNODES ${NNODES} x GRAD_ACCUM ${GRAD_ACCUM}"

torchrun --nnodes="${NNODES}" --nproc_per_node="${NPROC}" --node_rank="${NODE_RANK}" \
         --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  liquid/train/sft.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path "${BASE_CKPT}" \
    --version qwen \
    --data_path="${DATA_ROOT}/ai2d_tokenized_${TOKENIZER}^^${DATA_ROOT}/dvqa_tokenized_${TOKENIZER}^^${DATA_ROOT}/mgm_instruction_tokenized_${TOKENIZER}^^${DATA_ROOT}/lmsys_instruction_tokenized^^${DATA_ROOT}/llava_pretrain_tokenized_${TOKENIZER}^^${DATA_ROOT}/allava_laion_caption_tokenized_${TOKENIZER}^^${DATA_ROOT}/allava_vflan_caption_tokenized_${TOKENIZER}^^${DATA_ROOT}/laion_sft_tokenized_${TOKENIZER}_hf^^${DATA_ROOT}/journeydb_sft_tokenized_${TOKENIZER}_hf^^${DATA_ROOT}/blip3o_60k_sft_${TOKENIZER}_hf" \
    --t2i_source laion-aesthetics^^midjourney^^blip3o-60k \
    --shuffleseed 37 \
    --percentage 1.0^^1.0^^1.0^^1.0^^1.0^^1.0^^1.0^^0.94^^0.94^^1.0 \
    --vq_resolution 256 \
    --bf16 \
    --output_dir "${CKPT_ROOT}/${EXP_NAME}" \
    --run_name "${EXP_NAME}" \
    --num_train_epochs 2 \
    --per_device_train_batch_size "${BS}" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --save_strategy "no" \
    --learning_rate 5e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb
