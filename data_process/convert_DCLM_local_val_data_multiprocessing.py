import os
import zstandard as zstd
from tqdm import tqdm
import io
import json
import argparse
import subprocess
import boto3
import tarfile
import webdataset as wds
import multiprocessing as mp
import random
from datasets import load_dataset

def main(args):
    print("\nWebDataset loading...")
    s3 = boto3.client('s3')
    bucket = args.data_path.split("/")[2]
    prefix = "/".join(args.data_path.split("/")[3:])
    paginator = s3.get_paginator("list_objects_v2")
    tar_files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".tar"):
                tar_files.append(obj["Key"])

    # for key in tar_files:
    #     if "//" in key:
    #         print(f"Warning: skipping invalid key {key}")
            
    urls = [f"pipe:aws s3 cp s3://{bucket}/{key} -" for key in tar_files if "//" not in key]
    rng = random.Random(args.shuffleseed)
    rng.shuffle(urls)
    print(f"Total {len(urls)} tar files found.")
    urls = urls[-int(len(urls) * 0.02 * 0.002):]  # 0.015
    dataset = wds.WebDataset(urls, shardshuffle=args.shuffleseed)
    all_datas = []

    total_cnt = 0

    os.makedirs(args.temp_path, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    for sample in dataset:
        total_cnt += 1
        if isinstance(sample["txt"], bytes):
            text = sample["txt"].decode("utf-8")
        else:
            text = sample["txt"]  # Already decoded
            
        if isinstance(sample["json"], bytes):
            metadata_str = sample["json"].decode("utf-8")
        else:
            metadata_str = sample["json"]
            
        metadata = json.loads(metadata_str)
        
        all_datas.append(
            {    
                'data_type': 'text_pretrain',
                'text': text,
                'length': len(text),
                'vqcode_512': metadata.get('vqcode_512', 'no'),
                'vqcode_multi768': metadata.get('vqcode_multi768', 'no'),
                'width': metadata.get('width', 'no'),
                'height': metadata.get('height', 'no'),
            }
        )
        if total_cnt % 10000 == 0:
            with open(  os.path.join(args.temp_path, str(total_cnt//10000).zfill(6)+'.jsonl'  ), 'w') as f:
                for item in all_datas:
                    f.write(json.dumps(item)+'\n')
            all_datas = []
            print(f"Processed {total_cnt} samples")

        if total_cnt >= 50000:  # 50K
            break

    file_list = os.listdir(args.temp_path)
    data_list = [args.temp_path+"/"+filename for filename in file_list ]
    data_files = {"val":data_list }
    my_dataset = load_dataset("json", data_files=data_files, split='val', streaming=False, num_proc=40)
 
    my_dataset = my_dataset.shuffle(seed=42)
    my_dataset.save_to_disk(args.save_path, num_shards=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCLM to WebDataset using multiprocessing")
    parser.add_argument('--data-path', type=str, 
                       default='s3://your-s3-bucket/datasets/dclm_webdataset',
                       help='S3 path to save WebDataset files')
    parser.add_argument("--shuffleseed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument('--temp-path', type=str, default='/path/to/save/tempdata',help='path to save converted jsonl files')
    parser.add_argument('--save-path', type=str, default='/path/to/save/hfdata',help='path to save converted hf files')
    args = parser.parse_args()
    main(args)

# python convert_DCLM_local_val_data_multiprocessing.py --temp-path /tmp/dclm30m --save-path /path/to/data/dclm30m_val_hf