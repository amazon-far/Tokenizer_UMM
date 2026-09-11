# Data Processing After Tokenization

For tokenization, you can refer to `laion_tokenize_unitok_exec.yaml`. After tokenizing LAION-Aesthetics, JourneyDB, BLIP3o-short, BLIP3o-60k, and other SFT data, please follow these steps:

```bash
# LAION-Aesthetics
export TOKENIZER_NAME=your_tokenizer_name
cd /path/to/Tokenizer_UMM/data_process/
mkdir /path/to/data/laion_tokenized_$TOKENIZER_NAME
mv /path/to/data/laion_part_{1..12}_tokenized_$TOKENIZER_NAME /path/to/data/laion_tokenized_$TOKENIZER_NAME/
python laion_movejsonl.py --folder_path /path/to/data/laion_tokenized_$TOKENIZER_NAME
python jsonl_to_arrow.py --folder_path /path/to/data/laion_tokenized_$TOKENIZER_NAME --save_path /path/to/data/laion_tokenized_$TOKENIZER_NAME\_hf --num_shards 128
rm -rf /path/to/data/laion_tokenized_$TOKENIZER_NAME

python laion_movejsonl.py --folder_path /path/to/data/laion_val_tokenized_$TOKENIZER_NAME
python jsonl_to_arrow.py --folder_path /path/to/data/laion_val_tokenized_$TOKENIZER_NAME --save_path /path/to/data/laion_val_tokenized_$TOKENIZER_NAME\_hf --num_shards 1
rm -rf /path/to/data/laion_val_tokenized_$TOKENIZER_NAME

python tokenized_split.py --folder_path /path/to/data/laion_tokenized_$TOKENIZER_NAME\_hf --val_save_path /path/to/data/laion_sft_tokenized_$TOKENIZER_NAME\_hf --val_sample 917427
python pt_to_sft.py --folder_path /path/to/data/laion_sft_tokenized_$TOKENIZER_NAME\_hf --save_path /path/to/data/laion_sft_tokenized_$TOKENIZER_NAME\_hf_modified 
rm -rf /path/to/data/laion_sft_tokenized_$TOKENIZER_NAME\_hf
mv /path/to/data/laion_sft_tokenized_$TOKENIZER_NAME\_hf_modified /path/to/data/laion_sft_tokenized_$TOKENIZER_NAME\_hf
```

```bash
# JourneyDB
export TOKENIZER_NAME=your_tokenizer_name
python journeydb_movejsonl.py --folder_path /path/to/data/journeydb_tokenized_$TOKENIZER_NAME
python jsonl_to_arrow.py --folder_path /path/to/data/journeydb_tokenized_$TOKENIZER_NAME --save_path /path/to/data/journeydb_tokenized_$TOKENIZER_NAME\_hf --num_shards 10
rm -rf /path/to/data/journeydb_tokenized_$TOKENIZER_NAME

python journeydb_movejsonl.py --folder_path /path/to/data/journeydb_val_tokenized_$TOKENIZER_NAME
python jsonl_to_arrow.py --folder_path /path/to/data/journeydb_val_tokenized_$TOKENIZER_NAME --save_path /path/to/data/journeydb_val_tokenized_$TOKENIZER_NAME\_hf --num_shards 1
rm -rf /path/to/data/journeydb_val_tokenized_$TOKENIZER_NAME

python tokenized_split.py --folder_path /path/to/data/journeydb_val_tokenized_$TOKENIZER_NAME\_hf --train_save_path /path/to/data/journeydb_val_tokenized_$TOKENIZER_NAME\_hf_modified --val_save_path /path/to/data/journeydb_sft_tokenized_$TOKENIZER_NAME\_hf --val_sample 81319
rm -rf /path/to/data/journeydb_val_tokenized_$TOKENIZER_NAME\_hf
mv /path/to/data/journeydb_val_tokenized_$TOKENIZER_NAME\_hf_modified /path/to/data/journeydb_val_tokenized_$TOKENIZER_NAME\_hf
python pt_to_sft.py --folder_path /path/to/data/journeydb_sft_tokenized_$TOKENIZER_NAME\_hf --save_path /path/to/data/journeydb_sft_tokenized_$TOKENIZER_NAME\_hf_modified 
rm -rf /path/to/data/journeydb_sft_tokenized_$TOKENIZER_NAME\_hf
mv /path/to/data/journeydb_sft_tokenized_$TOKENIZER_NAME\_hf_modified /path/to/data/journeydb_sft_tokenized_$TOKENIZER_NAME\_hf
```

```bash
# BLIP3o-short
python tokenized_split.py --folder_path /path/to/data/blip3o_60k_short_tokenized_$TOKENIZER_NAME --train_save_path /path/to/data/blip3o_short_tokenized_$TOKENIZER_NAME\_hf --val_save_path /path/to/data/blip3o_val_short_tokenized_$TOKENIZER_NAME\_hf --val_sample 9560
rm -rf /path/to/data/blip3o_60k_short_tokenized_$TOKENIZER_NAME

# BLIP3o-60k
python pt_to_sft.py --folder_path /path/to/data/blip3o_60k_tokenized_$TOKENIZER_NAME --save_path /path/to/data/blip3o_60k_sft_$TOKENIZER_NAME\_hf
rm -rf /path/to/data/blip3o_60k_tokenized_$TOKENIZER_NAME
```