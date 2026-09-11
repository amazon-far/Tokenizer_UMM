#!/bin/bash
# Qwen3-4B-Base annealing / decay phase.
# Thin wrapper over qwen3_0_6b_mixpretrain_anneal.sh (single source of truth for the recipe).
export SIZE="4b"
export MODEL_NAME="${MODEL_NAME:-Qwen3-4B-Base}"
exec bash "$(dirname "$0")/qwen3_0_6b_mixpretrain_anneal.sh"
