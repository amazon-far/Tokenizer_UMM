#!/bin/bash
# Build the conda environments used for training and evaluation:
#   - liquid : training, data tokenization, and most evaluation (bpb, VQA, generation)
#   - t2v    : VQAScore scoring for GenAI-Bench (evaluation/T2I_Eval/eval_genai_527.py)
#   - base   : OpenAI client for WISE scoring (evaluation/T2I_Eval/eval_wise.py)
#
# All image tokenizers (GigaTok, UniTok, IBQ / SEED-Voken, Chameleon) run inside
# the `liquid` env — they need only three extra packages on top of it, installed
# in section 1 below. The upstream projects ship their own requirements.txt
# pinning different torch / transformers versions, but none of those pins are
# needed to *encode* images; installing them would only downgrade `liquid`.
#
# NOT built here: the `dpg` env used by evaluation/T2I_Eval/eval_dpg.sh for
# mPLUG-based DPG-Bench scoring — its dependencies conflict with `liquid`.
# Create it separately if you need DPG-Bench (see evaluation/EVAL.md#dpg-bench).
#
# Usage:
#   bash scripts/setup_env.sh            # build all three envs
#   conda activate liquid
set -ex

source "$(conda info --base)/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# 1) liquid : main training / tokenization / evaluation environment
# ---------------------------------------------------------------------------
ENV_NAME="${ENV_NAME:-liquid}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
conda activate "${ENV_NAME}"

pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
# flash-attention (built against the installed torch)
pip install flash-attn==2.5.8 --no-build-isolation
# data / logging / IO utilities used by the data pipeline and trainer
pip install datasets zstandard boto3 s3fs webdataset wandb
# transformers version used for the experiments
pip install transformers==4.51.0
# tokenizer backbones (GigaTok / UniTok use timm) and plotting
pip install timm==1.0.8          # newer (e.g. 1.0.22) also works
pip install matplotlib
# the only extra packages the image tokenizers need on top of `liquid`:
#   omegaconf                     -> IBQ config loading (src/IBQ/models/ibqgan.py)
#   pytorch-lightning, lightning  -> IBQ model base classes (same file)
# GigaTok, UniTok and Chameleon need nothing beyond `liquid` itself. None of
# these three touch torch / transformers / numpy, so the env is unchanged
# otherwise (pytorch-lightning also pulls in torchmetrics + lightning-utilities).
pip install omegaconf pytorch-lightning lightning
# evaluation helpers: LM-eval harness, MJHQ FID, misc parsing
pip install lm-eval
# lm-eval pins `datasets`; re-upgrade so the data pipeline keeps a recent version
pip install -U datasets
pip install clean-fid scipy==1.11.1 typed-argument-parser ftfy

conda deactivate

# ---------------------------------------------------------------------------
# 2) t2v : VQAScore scoring for GenAI-Bench (skip if not scoring GenAI-Bench)
#          See https://github.com/linzhiqiu/t2v_metrics
# ---------------------------------------------------------------------------
conda create -n t2v python=3.10 -y
conda activate t2v
conda install pip -y
conda install -c conda-forge ffmpeg=7 -y
pip install torch torchvision torchaudio
pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/linzhiqiu/pytorchvideo.git
pip install t2v-metrics
# prebuilt flash-attn wheel matching this env's torch/cuda — adjust the URL if
# your torch/cuda/python differ (see the flash-attention releases page).
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
conda deactivate

# ---------------------------------------------------------------------------
# 3) base : OpenAI client for WISE scoring. Set OPENAI_API_KEY before running WISE.
# ---------------------------------------------------------------------------
conda activate base
pip install openai==0.28.0
conda deactivate

set +x
echo "Environments ready:"
echo "  liquid  -> training, data tokenization (all image tokenizers), bpb/VQA/generation eval"
echo "  t2v     -> GenAI-Bench VQAScore scoring"
echo "  base    -> +openai for WISE scoring (export OPENAI_API_KEY)"
echo "Tokenizer weights are not installed here — see evaluation/TOKENIZERS.md."
