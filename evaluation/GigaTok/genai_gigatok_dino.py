# from huggingface_hub import snapshot_download
# snapshot_download(repo_id='BLIP3o/BLIP3o-Pretrain-Short-Caption', repo_type='dataset', local_dir='/path/to/data/BLIP3o_60k_short')

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
from torchvision import transforms
from datasets import load_dataset
from utils.model_init import load_model_from_config, custom_load
from dataset.augmentation import center_crop_arr
import yaml
import torch.nn.functional as F
ImageFile.LOAD_TRUNCATED_IMAGES = False
torch.set_grad_enabled(False)

def process_item(batch,image_tokenizer,device):
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
                image = Image.open(image_data)
            except Exception as e:
                print(e)
                print('load error', image_data)
                continue
        elif hasattr(image_data, 'convert'):
            # It's already a PIL Image
            image = image_data
        try:
            ori_image = image.convert("RGB")
            # Process image
            transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
            input_image = transform(ori_image)
        except Exception as e:
            print('crop error', e)
            continue
        
        input_images.append(input_image)
        new_batch.append(data)
            
    if len(input_images) > 0:
        input_images = torch.stack(input_images, dim=0).to(device)
        with torch.no_grad():
            latent, _, [_, _, indices] = image_tokenizer.encode(input_images)
            vqcodes = indices.cpu().view(len(input_images), -1).tolist()
        

    results = []
    for idx, data in enumerate(new_batch):
        caption = data['txt']
        new_anno = {
            "tokenized": True,
            "source": "{\"images\": \"" + data['source'] + "\", \"captions\": \"GenAI-Bench-527\"}",   
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

def load_prompts_as_dataset(path: str) -> Dataset:
    source_names = ["DALLE_3", "DeepFloyd_I_XL_v1", "Midjourney_6", "SDXL_2_1", "SDXL_Base", "SDXL_Turbo"]
    dataset_dict = {"txt": [], "jpg": [], "source": []}
    with open(os.path.join(path, "prompts.txt"), "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
        for source_name in source_names:
            dataset_dict["txt"].extend(prompts)
            dataset_dict["jpg"].extend([os.path.join(path, source_name, f"{i+1:05d}.jpeg") for i in range(len(prompts))])
            dataset_dict["source"].extend([source_name] * len(prompts))
    return Dataset.from_dict(dataset_dict)

def worker_process(rank, args, dataset_path, world_size):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    model_config = "configs/vq/VQ_BL256_dino_disc.yaml"
    with open(model_config, "r") as f:
        config = yaml.safe_load(f)
    image_tokenizer = load_model_from_config(config)
    causal_type = config["model"]["causal_settings"]["causal_type"]
    image_tokenizer.eval()
    image_tokenizer.requires_grad_(False)
    image_tokenizer = image_tokenizer.to(torch.device(f"cuda:{rank}"))
    print(f"VQ Model Parameters(inference): {sum(p.numel() for p in image_tokenizer.parameters()):,}")
    ckpt_path = "VQ_BL256_dino_disc.pt"

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "ema" in checkpoint:  # ema
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")

    # image_tokenizer.load_state_dict(model_weight)
    custom_load(image_tokenizer, model_weight)
    del checkpoint
    # Load dataset in this worker
    dataset = load_prompts_as_dataset(args.input_pairs)
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
        
        batch_results = process_item(batch, image_tokenizer, device=device)
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
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for each process')
    
    args = parser.parse_args()
    main(args)

# python genai_gigatok_dino.py --input_pairs /path/to/data/siting/GenAI-Bench-527 --save_path /path/to/data/GenAI-Bench-527_tokenized_gigatok_dino --temp_path /tmp/genai_bench_527_jsonl_gigatok_dino 