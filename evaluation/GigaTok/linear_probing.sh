. set_env_vars.sh
export TOK_CONFIG="configs/vq/VQ_BL256_dino_disc.yaml"
export VQ_CKPT="VQ_BL256_dino_disc.pt"
# the precision of the tokenizer
export TOK_PRECISION="none" # fp32

# directory for saving the linear probing results
export SAVE_DIR="results/lin_probe"
# the final results are in ${SAVE_DIR}/${LM_LIN_EXP_DIR}
export TOK_LIN_EXP_DIR=lin_VQ_BL256_dino_disc
# the batch size of the linear probing training
export LIN_BSZ=128

bash scripts/composite_cmd/train_tok_lin_probe_and_eval.sh
