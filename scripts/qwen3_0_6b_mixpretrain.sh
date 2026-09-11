#!/bin/bash
# Continual pre-training (stage 1): unified AR mixed-modal training.
#
# Extends the Qwen3 base model's vocabulary with image tokens, then trains on a
# mix of pure-text (DCLM) and image-text (LAION / JourneyDB / BLIP3o) data that
# has been pre-tokenized with the chosen image tokenizer.
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
SIZE="${SIZE:-0.6b}"                              # model size tag used in EXP_NAME
MODEL_NAME="${MODEL_NAME:-Qwen3-0.6B-Base}"       # HF base LLM directory name under $DATA_ROOT
BASE_MODEL="${BASE_MODEL:-${DATA_ROOT}/${MODEL_NAME}}"
TOKENIZER="${TOKENIZER:-gigatok}"    # gigatok | gigatok_dino | ibq_1024 | ibq_8192 | ibq_16384 | unitok | unitok_sem
VOCAB="${VOCAB:-16384}"              # image codebook size; must match TOKENIZER (e.g. ibq_1024 -> 1024)
LR="${LR:-3e-5}"
BS="${BS:-16}"                       # per-device batch size
EXP_NAME="${EXP_NAME:-${TOKENIZER}_qwen3_${SIZE}_mixpretrain}"

# ---- distributed launch (defaults to a single 8-GPU node) ----
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-2935}"
NPROC="${NPROC:-8}"

# ---- global batch size ----
# The recipe is defined by the GLOBAL batch, so gradient accumulation is derived
# from however many GPUs you actually have: BS x NPROC x NNODES x GRAD_ACCUM.
GLOBAL_BS="${GLOBAL_BS:-512}"
_per_step=$(( BS * NPROC * NNODES ))
if [ $(( GLOBAL_BS % _per_step )) -ne 0 ]; then
    echo "ERROR: GLOBAL_BS=${GLOBAL_BS} is not divisible by BS*NPROC*NNODES=${_per_step}." >&2
    echo "       Adjust BS, NPROC, NNODES or GLOBAL_BS so they divide evenly." >&2
    exit 1
fi
GRAD_ACCUM=$(( GLOBAL_BS / _per_step ))
echo "global batch ${GLOBAL_BS} = BS ${BS} x NPROC ${NPROC} x NNODES ${NNODES} x GRAD_ACCUM ${GRAD_ACCUM}"

# 1) Expand the LLM vocabulary / LM head to model VOCAB image tokens.
ADDTOKEN_MODEL="${BASE_MODEL}-addtoken"
python liquid/expand_vocabulary.py \
    --model_path "${BASE_MODEL}" \
    --save_path "${ADDTOKEN_MODEL}" \
    --num_add_token "${VOCAB}"

# 2) Continual pre-training. Data paths point at datasets pre-tokenized with the
#    chosen tokenizer (see data_process/ and evaluation/<tokenizer>/tokenize_*).
#    `--percentage` sets the sampling proportion of each source; `--T2I_ratio`
#    is the fraction of image-text data formatted in text-to-image order.
#    JourneyDB is listed twice on purpose: the mixture samples it at 1.0 + 0.563
#    (~1.563 epochs) while every other image-text source is used once. Keep the
#    two entries and their ratio in sync if you rescale the mixture.
torchrun --nnodes="${NNODES}" --nproc_per_node="${NPROC}" --node_rank="${NODE_RANK}" \
         --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  liquid/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path "${ADDTOKEN_MODEL}" \
    --version qwen \
    --data_path="${DATA_ROOT}/dclm30m_hf^^${DATA_ROOT}/laion_tokenized_${TOKENIZER}_hf^^${DATA_ROOT}/journeydb_tokenized_${TOKENIZER}_hf^^${DATA_ROOT}/journeydb_tokenized_${TOKENIZER}_hf^^${DATA_ROOT}/blip3o_short_tokenized_${TOKENIZER}_hf" \
    --shuffleseed 42 \
    --percentage 0.2222^^1.0^^1.0^^0.563^^1.0 \
    --T2I_ratio 0.8 \
    --vq_resolution 256 \
    --bf16 \
    --output_dir "${CKPT_ROOT}/${EXP_NAME}" \
    --run_name "${EXP_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size "${BS}" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --save_strategy "steps" \
    --save_steps 0.1 \
    --save_total_limit 10 \
    --learning_rate "${LR}" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb
