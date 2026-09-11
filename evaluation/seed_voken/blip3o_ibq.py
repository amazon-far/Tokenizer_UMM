# from huggingface_hub import snapshot_download
# snapshot_download(repo_id='BLIP3o/BLIP3o-Pretrain-Short-Caption', repo_type='dataset', local_dir='/path/to/data/BLIP3o_60k_short')

# --- path configuration (see .env.example) ---
import os
HF_CACHE = os.environ.get("HF_CACHE", "/path/to/hf_cache")

import os
import json
import argparse
import PIL
import glob
from PIL import Image, ImageFile
import argparse
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np
import datasets
from datasets import Dataset
from datasets import load_dataset
from omegaconf import OmegaConf
from src.IBQ.models.ibqgan import IBQ

import yaml
import torch.nn.functional as F
ImageFile.LOAD_TRUNCATED_IMAGES = False
torch.set_grad_enabled(False)

def center_crop_image(ori_image, tgt_width=256, tgt_height=256):
    Width,Height = ori_image.size
    factor = min(Width,Height)/min(tgt_width,tgt_height)
    input_image = ori_image.resize((int(Width/factor),int(Height/factor)), PIL.Image.LANCZOS)
    resize_width, resize_height = input_image.size   # Get dimensions

    left = (resize_width - tgt_width)//2
    top = (resize_height - tgt_height)//2
    right = (resize_width + tgt_width)//2
    bottom = (resize_height + tgt_height)//2
    # Crop the center of the image
    input_image = input_image.crop((left, top, right, bottom))
    return input_image

def process_item(batch,image_tokenizer,source,image_folder,device):
    input_images = []
    new_batch = []
    for data in batch:
        image_data = data['jpg']

        # Check if it's a dict (HF Image format)
        if isinstance(image_data, dict):
            if image_data.get('bytes') is not None:
                # Load from bytes
                from io import BytesIO
                image = Image.open(BytesIO(image_data['bytes']))
            elif image_data.get('path') is not None:
                # Load from path
                image = Image.open(image_data['path'])
            else:
                # Skip if no valid image data
                continue
        elif isinstance(image_data, str):
            # It's a path string
            try:
                image = Image.open(os.path.join(image_folder, image_data))
            except:
                print('load error', image_data)
                continue
        elif hasattr(image_data, 'convert'):
            # It's already a PIL Image
            image = image_data
        try:
            ori_image = image.convert("RGB")
            # Process image
            image = center_crop_image(ori_image)
            image = np.array(image).astype(np.uint8)
            image = (image/127.5 - 1.0).astype(np.float32)
            x = torch.tensor(image)
            x = torch.einsum('hwc->chw', x)
            x = x.float()
        except Exception as e:
            print('crop error', e)
            continue
        
        input_images.append(x)
        new_batch.append(data)
            
    if len(input_images) > 0:
        input_images = torch.stack(input_images, dim=0).to(device)
        with torch.no_grad():
            quant, qloss, (_, _, indices) = image_tokenizer.encode(input_images)
            vqcodes = indices.cpu().view(len(input_images), -1).tolist()
        

    results = []
    for idx, data in enumerate(new_batch):
        caption = data['txt']
        new_anno = {
            "tokenized": True,
            "source": "{\"images\": \"" + source + "\", \"captions\": \"internvl3_1b\"}",   
            'data_type': 'image_text',
            'text': caption,
            'length': len(caption) + len(vqcodes[idx]) * 4,
            'vqcode_256': json.dumps(vqcodes[idx]),
            'vqcode_512': 'no',
            'vqcode_multi768': 'no',
            'width': 'no',
            'height': 'no',
        }
        results.append(new_anno)

    return results


def worker_process(rank, args, dataset_path, world_size):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    configs = OmegaConf.load(f"configs/IBQ/gpu/imagenet_ibqgan_{args.vocab_size}.yaml")
    encoder = IBQ(**configs.model.init_args)
    sd = torch.load(f"imagenet256_{args.vocab_size}.ckpt", map_location="cpu")["state_dict"]
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    encoder.eval()
    encoder.requires_grad_(False)
    encoder = encoder.to(torch.device(f"cuda:{rank}"))

    # Load dataset in this worker
    data_files = glob.glob(os.path.join(args.input_pairs, '*.tar')) 
    dataset = load_dataset("webdataset", data_files=data_files, cache_dir=HF_CACHE + '/', split="train", num_proc=64)
    subset = dataset.shard(num_shards=world_size, index=rank)
    print(f"GPU {rank}: Processing {len(subset)} items")
    
    # Use ThreadPoolExecutor instead of Pool for better GPU sharing
    valid_pair_list = []
    for start_index in tqdm(range(0, len(subset), args.batch_size), desc=f"GPU {rank} processing"):
        end_index = min(start_index + args.batch_size, len(subset))
        
        # Get batch without converting entire dataset to list
        if isinstance(subset, list):
            batch = subset[start_index:end_index]
        else:
            # For HuggingFace dataset, use select for efficient indexing
            batch_indices = list(range(start_index, end_index))
            batch = [subset[i] for i in batch_indices]
        
        batch_results = process_item(batch, encoder, args.source_name, image_folder=None, device=device)
        valid_pair_list.extend(batch_results)
        
    valid_images = len(valid_pair_list)
    print(f'GPU {rank}: Kept {valid_images} items from {len(subset)} total')

    # Save results for this rank
    if not os.path.exists(args.temp_path):
        os.makedirs(args.temp_path)

    output_file = os.path.join(args.temp_path, f'gpu_{rank:03d}.jsonl')
    with open(output_file, 'w') as f:
        for item in valid_pair_list:
            f.write(json.dumps(item)+'\n')

def main(args):
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for distributed processing")
    
    if world_size <= 1:
        print("Warning: Only 1 GPU available, falling back to single GPU processing")
        worker_process(0, args, args.input_pairs, world_size)
    else:
        # Pass the dataset path instead of the data itself
        mp.spawn(worker_process, args=(args, args.input_pairs, world_size), nprocs=world_size, join=True)
    
    # Find all GPU result files
    gpu_files = []
    for i in range(world_size):
        gpu_file = os.path.join(args.temp_path, f'gpu_{i:03d}.jsonl')
        if os.path.exists(gpu_file):
            gpu_files.append(gpu_file)
    
    print(f"Found {len(gpu_files)} GPU result files")
    
    # Create dataset directly from all files
    if gpu_files:
        all_data = []
        for gpu_file in gpu_files:
            print(f"Loading data from {gpu_file}")
            with open(gpu_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        item = json.loads(line)
                        all_data.append(item)
        
        # Save merged dataset
        ds = Dataset.from_list(all_data)
        os.makedirs(args.save_path, exist_ok=True)
        ds.save_to_disk(args.save_path)
        
        print(f"HuggingFace dataset with {len(ds)} items saved to {args.save_path}")
    else:
        print("No GPU result files found!")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_pairs', type=str, help='jsonl file, where image pairs meta saved')
    parser.add_argument('--temp_path', type=str, help='path to save converted jsonl files')
    parser.add_argument('--save_path', type=str, help='path to save converted hf datasets files')
    parser.add_argument('--source_name', type=str, help='source name to tag the dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for each process')
    parser.add_argument('--vocab_size', type=int, required=True, help='vocabulary size for IBQ tokenizer')
    
    args = parser.parse_args()
    main(args)

# python blip3o_ibq.py --input_pairs /path/to/data/BLIP3o_60k --save_path /path/to/data/blip3o_60k_tokenized_ibq_16384 --temp_path /tmp/blip3o_60k_jsonl_ibq_16384 --source_name blip3o-60k --vocab_size 16384
# python blip3o_ibq.py --input_pairs /path/to/data/BLIP3o_60k_short --save_path /path/to/data/blip3o_60k_short_tokenized_ibq_16384 --temp_path /tmp/blip3o_60k_short_jsonl_ibq_16384 --source_name blip3o-60k-short --vocab_size 16384