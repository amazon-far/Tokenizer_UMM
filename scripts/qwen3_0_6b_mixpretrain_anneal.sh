#!/bin/bash
# Continual pre-training — annealing (decay) phase of the WSD schedule.
#
# Resumes from a stage-1 "stable-phase" checkpoint and linearly decays the
# learning rate over a short annealing run. It runs in the SAME output
# directory as the stage-1 run (scripts/qwen3_0_6b_mixpretrain.sh) and resumes
# from ${STABLE_CKPT}; rename_ckpt.py sets the resume point before training and
# marks the final annealed checkpoint (acheckpoint-*) afterwards.
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
MODEL_NAME="${MODEL_NAME:-Qwen3-0.6B-Base}"
BASE_MODEL="${BASE_MODEL:-${DATA_ROOT}/${MODEL_NAME}}"
TOKENIZER="${TOKENIZER:-gigatok}"
VOCAB="${VOCAB:-16384}"
LR="${LR:-3e-5}"
BS="${BS:-16}"
EXP_NAME="${EXP_NAME:-${TOKENIZER}_qwen3_${SIZE}_mixpretrain}"  # stage-1 run dir (annealing continues it)
STABLE_CKPT="${STABLE_CKPT:-checkpoint-23438}"              # stable-phase checkpoint to anneal from

# Annealing data amount. MUST be scaled to STABLE_CKPT: size it so this run's
# total steps ~= 1.25 x (STABLE_CKPT global step), since the decay is the last
# 1/5 of the run and --warmup_ratio 0.8 places the LR peak at the resume point.
# Per-source fractions, `^^`-joined (matches the --data_path order below, in
# which JourneyDB appears twice — see qwen3_0_6b_mixpretrain.sh).
ANNEAL_PERCENTAGE="${ANNEAL_PERCENTAGE:-0.05555^^0.25^^0.25^^0.14075^^0.25}"

# ---- distributed launch (defaults to a single 8-GPU node) ----
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-2935}"
NPROC="${NPROC:-8}"

# ---- global batch size (must match the stage-1 run this anneals from) ----
GLOBAL_BS="${GLOBAL_BS:-512}"
_per_step=$(( BS * NPROC * NNODES ))
if [ $(( GLOBAL_BS % _per_step )) -ne 0 ]; then
    echo "ERROR: GLOBAL_BS=${GLOBAL_BS} is not divisible by BS*NPROC*NNODES=${_per_step}." >&2
    echo "       Adjust BS, NPROC, NNODES or GLOBAL_BS so they divide evenly." >&2
    exit 1
fi
GRAD_ACCUM=$(( GLOBAL_BS / _per_step ))
echo "global batch ${GLOBAL_BS} = BS ${BS} x NPROC ${NPROC} x NNODES ${NNODES} x GRAD_ACCUM ${GRAD_ACCUM}"

ADDTOKEN_MODEL="${BASE_MODEL}-addtoken"       # created in stage 1
OUTPUT_DIR="${CKPT_ROOT}/${EXP_NAME}"

# On rank 0: keep ${STABLE_CKPT} as the resume head (rename the rest to lcheckpoint*).
if [ "${NODE_RANK}" == "0" ]; then
    python rename_ckpt.py --base "${OUTPUT_DIR}/" --head "${STABLE_CKPT}"
fi

# Annealing run: linear LR decay (WSD decay phase), resumes from ${STABLE_CKPT}.
torchrun --nnodes="${NNODES}" --nproc_per_node="${NPROC}" --node_rank="${NODE_RANK}" \
         --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  liquid/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path "${ADDTOKEN_MODEL}" \
    --version qwen \
    --data_path="${DATA_ROOT}/dclm30m_hf^^${DATA_ROOT}/laion_tokenized_${TOKENIZER}_hf^^${DATA_ROOT}/journeydb_tokenized_${TOKENIZER}_hf^^${DATA_ROOT}/journeydb_tokenized_${TOKENIZER}_hf^^${DATA_ROOT}/blip3o_short_tokenized_${TOKENIZER}_hf" \
    --shuffleseed 42 \
    --percentage "${ANNEAL_PERCENTAGE}" \
    --T2I_ratio 0.8 \
    --vq_resolution 256 \
    --bf16 \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${EXP_NAME}_anneal" \
    --num_train_epochs 1 \
    --per_device_train_batch_size "${BS}" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --save_strategy "no" \
    --learning_rate "${LR}" \
    --weight_decay 0.0 \
    --warmup_ratio 0.8 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb

# On rank 0: mark the final annealed checkpoint as acheckpoint-*.
if [ "${NODE_RANK}" == "0" ]; then
    python rename_ckpt.py --base "${OUTPUT_DIR}/" --target "a${STABLE_CKPT}"
fi
