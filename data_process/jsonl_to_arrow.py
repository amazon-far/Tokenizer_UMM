import os
import json
import argparse

from datasets import load_dataset

def main(args):
    my_dataset = load_dataset("json", data_dir=args.folder_path, split='train',streaming=False, num_proc=40)
 
    my_dataset = my_dataset.shuffle(seed=42)
    my_dataset.save_to_disk(args.save_path, num_shards=args.num_shards)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--folder_path', type=str, default='/path/to/jsonl/files',help='path to folder containing jsonl files')
    parser.add_argument('--save_path', type=str, default='/path/to/save/hfdata',help='path to save converted hf datasets files')
    parser.add_argument('--num_shards', type=int, default=128, help='number of shards to split the dataset into')
    args = parser.parse_args()
    main(args)

# python jsonl_to_arrow.py --folder_path /path/to/data/laion_added_tokenized_gigatok --save_path /path/to/data/laion_added_tokenized_gigatok_hf --num_shards 128
# python jsonl_to_arrow.py --folder_path /path/to/data/laion_tokenized_gigatok --save_path /path/to/data/laion_tokenized_gigatok_hf --num_shards 128
# python jsonl_to_arrow.py --folder_path /path/to/data/journeydb_tokenized_gigatok --save_path /path/to/data/journeydb_tokenized_gigatok_hf --num_shards 10
# python jsonl_to_arrow.py --folder_path /path/to/data/journeydb_val_tokenized_gigatok --save_path /path/to/data/journeydb_val_tokenized_gigatok_hf --num_shards 1
# rm -rf /path/to/data/laion_added_tokenized_gigatok
# rm -rf /path/to/data/laion_tokenized_gigatok
# rm -rf /path/to/data/journeydb_tokenized_gigatok
# rm -rf /path/to/data/journeydb_val_tokenized_gigatok
# 10 for journeydb