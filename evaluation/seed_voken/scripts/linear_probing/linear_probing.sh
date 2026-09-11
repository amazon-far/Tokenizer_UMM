. set_env_vars.sh
export TOK_CONFIG="../../configs/IBQ/gpu/imagenet_ibqgan_16384.yaml"
export VQ_CKPT="../../imagenet256_16384.ckpt"
# the precision of the tokenizer
export TOK_PRECISION="none" # fp32

# directory for saving the linear probing results
export SAVE_DIR="results/lin_probe"
# the final results are in ${SAVE_DIR}/${LM_LIN_EXP_DIR}
export TOK_LIN_EXP_DIR=lin_IBQ_imagenet_ibqgan_16384
# the batch size of the linear probing training
export LIN_BSZ=128

bash train_tok_lin_probe_and_eval.sh
