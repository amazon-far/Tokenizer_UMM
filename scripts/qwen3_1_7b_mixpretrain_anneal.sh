#!/bin/bash
# Qwen3-1.7B-Base annealing / decay phase.
# Thin wrapper over qwen3_0_6b_mixpretrain_anneal.sh (single source of truth for the recipe).
export SIZE="1.7b"
export MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B-Base}"
exec bash "$(dirname "$0")/qwen3_0_6b_mixpretrain_anneal.sh"
