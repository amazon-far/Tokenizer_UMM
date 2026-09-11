. set_env_vars.sh
export VQ_CKPT="/path/to/data/unitok_checkpoint_large_clip_1codebook_sem/ckpt-last.pth"
# the precision of the tokenizer
export TOK_PRECISION="none" # fp32

# directory for saving the linear probing results
export SAVE_DIR="results/lin_probe"
# the final results are in ${SAVE_DIR}/${LM_LIN_EXP_DIR}
export TOK_LIN_EXP_DIR=lin_UniTok_16384_sem_imagenet
# the batch size of the linear probing training
export LIN_BSZ=128

bash train_tok_lin_probe_and_eval.sh
