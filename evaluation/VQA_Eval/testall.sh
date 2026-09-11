chmod +x textvqa.sh pope.sh gqa.sh vqav2.sh mme.sh realworldqa.sh 


# export CKPT="gigatok_qwen3_0.6b_mixpretrain_stage2_1_8_0.8_60b_1e-4_2048_5860"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./pope.sh 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./gqa.sh 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./textvqa.sh 
./mme.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./vqav2.sh 
# ./mmvet.sh
# ./sqa.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./realworldqa.sh
