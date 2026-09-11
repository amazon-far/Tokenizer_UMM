#!/bin/bash
# Qwen3-1.7B-Base supervised fine-tuning.
# Thin wrapper over qwen3_0_6b_sft.sh (single source of truth for the recipe).
export SIZE="1.7b"
exec bash "$(dirname "$0")/qwen3_0_6b_sft.sh"
