
# --- path configuration (see .env.example) ---
import os
DATA_ROOT = os.environ.get("DATA_ROOT", "/path/to/data")
HF_CACHE = os.environ.get("HF_CACHE", "/path/to/hf_cache")

import datasets 
import os
import glob

from datasets import load_dataset

data_files = glob.glob(os.path.join(DATA_ROOT + "/BLIP3o_60k", '*.tar')) 

# STEP 1: check format
# sample = load_dataset("webdataset", data_files=[data_files[0]], cache_dir=HF_CACHE + '/', split="train", num_proc=1).select(range(1))
# print(sample.column_names)
# print(sample[0])

# STEP 2: convert to parquet
dataset = load_dataset("webdataset", data_files=data_files, cache_dir=HF_CACHE + '/', split="train", num_proc=64)
os.makedirs(DATA_ROOT + "/BLIP3o_60k_parquet", exist_ok=True)
dataset.to_parquet(DATA_ROOT + "/BLIP3o_60k_parquet/0.parquet")

# STEP 3: verify
# loaded_dataset = load_dataset("parquet", data_files=DATA_ROOT + "/BLIP3o_60k_parquet/0.parquet")
# sample = loaded_dataset['train'].select(range(1))
# print(sample.column_names)
# print(sample[0])