import os
from datasets import load_dataset
import PIL
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image, ImageFile
import torch
import io
import boto3
import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np
from omegaconf import OmegaConf
from src.IBQ.models.ibqgan import IBQ
import yaml
import argparse
from botocore.config import Config

BOTO3_CONFIG = Config(max_pool_connections=50, retries={'max_attempts': 5, 'mode': 'standard'})

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

def load_image_from_s3_uri(s3_uri, s3_client=None):
    """Load image from S3 URL"""
    try:
        # Parse S3 URL
        parts = s3_uri[5:].split('/', 1)
        bucket_name = parts[0]
        key = parts[1]
        
        if s3_client is None:
            s3_client = boto3.client('s3')
        
        # Download image from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        image_data = response['Body'].read()
        response['Body'].close()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image from {s3_uri}: {e}")
        return None


class ImageTextDataset(Dataset):
    """Dataset wrapper for streaming datasets"""
    def __init__(self, streaming_dataset, rank=0, world_size=1):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        print(f"GPU {rank}: Materializing dataset...")
        self.samples = []
        
        # Manually shard the dataset for this GPU
        for idx, sample in enumerate(streaming_dataset):
            if idx % world_size == rank:
                self.samples.append(sample)
        
        print(f"GPU {rank}: Dataset materialized with {len(self.samples)} samples")
        self._s3_client = None

    @property
    def s3_client(self):
        if self._s3_client is None:
            self._s3_client = boto3.client('s3', config=BOTO3_CONFIG)
        return self._s3_client
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load and preprocess image
            if sample['internvl3_1b_caption'] is None:
                return None
            
            ori_image = load_image_from_s3_uri(sample['s3_uri'], self.s3_client)
            
            if ori_image is None:
                return None
            
            # Check aspect ratio
            original_ratio = max(ori_image.size) / min(ori_image.size)
            if original_ratio > 2:
                return None
            
            # Process image
            image = center_crop_image(ori_image)
            image = np.array(image).astype(np.uint8)
            image = (image/127.5 - 1.0).astype(np.float32)
            x = torch.tensor(image)
            x = torch.einsum('hwc->chw', x)
            x = x.float()
            
            return {
                'image': x,
                'caption': sample['internvl3_1b_caption'],
                'valid': True
            }
        except Exception as e:
            print(f"Error processing image from {sample['s3_uri']}: {e}")
            return None

def worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    # ensure each worker re-creates its own client
    ds._s3_client = boto3.client('s3', config=BOTO3_CONFIG)

def collate_fn(batch):
    """Custom collate function to handle None values"""
    # Filter out None values
    batch = [item for item in batch if item is not None and item.get('valid', False)]
    
    if len(batch) == 0:
        return None
    
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    
    return {
        'images': images,
        'captions': captions
    }


def tokenize_batch(batch, image_tokenizer, gpu_id):
    """Tokenize a batch of samples and format according to new_anno structure"""
    if batch is None or len(batch['captions']) == 0:
        return []
    try:
        images = batch['images'].to(torch.device(f"cuda:{gpu_id}"))
        captions = batch['captions']
        
        with torch.no_grad():
            quant, qloss, (_, _, indices) = image_tokenizer.encode(images)
            vqcodes = indices.cpu().view(len(captions), -1).tolist()
        
        results = []
        for idx, caption in enumerate(captions):
            new_anno = {
                "tokenized": True,
                "source": "{\"images\": \"laion-aesthetics\", \"captions\": \"internvl3_1b\"}",
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
    
    except Exception as e:
        print(f"GPU {gpu_id}: Error processing batch: {e}")
        import traceback
        traceback.print_exc()
        return []

def process_on_gpu(gpu_id, world_size, folder_path, 
                   output_dir="tokenized_data",
                   batch_size=16,
                   save_every=500,
                   num_workers=4,
                   vocab_size=16384):
    """Process data on a single GPU - completely independent"""
    
    try:
        # Set device for this process
        torch.cuda.set_device(gpu_id)
        print(f"GPU {gpu_id}: Starting tokenization process")
        
        if os.path.isdir(folder_path):
            # If folder_path is a directory, find all .parquet files inside
            parquet_files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.endswith(".parquet")
            ]
            if not parquet_files:
                raise RuntimeError(f"No parquet files found in {folder_path}")
            streaming_dataset = load_dataset(
                "parquet",
                data_files=parquet_files,
                streaming=True,
                split="train"
            )
        else:
            # Assume folder_path is a single parquet file
            streaming_dataset = load_dataset(
                "parquet",
                data_files=folder_path,
                streaming=True,
                split="train"
            )
        
        # Create dataset with manual sharding
        dataset = ImageTextDataset(streaming_dataset, rank=gpu_id, world_size=world_size)
        
        # Simple DataLoader without DistributedSampler
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=1 if num_workers > 0 else None,
            worker_init_fn=worker_init_fn if num_workers > 0 else None
        )
        
        # Load model
        configs = OmegaConf.load(f"configs/IBQ/gpu/imagenet_ibqgan_{vocab_size}.yaml")
        encoder = IBQ(**configs.model.init_args)
        sd = torch.load(f"imagenet256_{vocab_size}.ckpt", map_location="cpu")["state_dict"]
        missing, unexpected = encoder.load_state_dict(sd, strict=False)
        encoder.eval()
        encoder.requires_grad_(False)
        encoder = encoder.to(torch.device(f"cuda:{gpu_id}"))

        # Process and save data
        output_path = f"{output_dir}/gpu_{gpu_id}"
        os.makedirs(output_path, exist_ok=True)

        batch_count = 0
        sample_count = 0
        processed_samples = []
        
        # Process dataset in batches
        print(f"GPU {gpu_id}: Starting to process samples...")
        for batch in tqdm(dataloader, desc=f"GPU {gpu_id}"):
            if batch is None:
                continue
            tokenized_samples = tokenize_batch(batch, encoder, gpu_id)
            processed_samples.extend(tokenized_samples)
            sample_count += len(tokenized_samples)
            
            # Save if we have enough samples
            if len(processed_samples) >= save_every:
                save_path = f"{output_path}/batch_{batch_count:06d}.jsonl"
                save_batch_jsonl(processed_samples, save_path)
                
                print(f"GPU {gpu_id}: Saved batch {batch_count}, total samples: {sample_count}")
                
                processed_samples = []
                batch_count += 1

        # Save remaining samples
        if processed_samples:
            save_path = f"{output_path}/batch_{batch_count:06d}.jsonl"
            save_batch_jsonl(processed_samples, save_path)
            print(f"GPU {gpu_id}: Saved final batch {batch_count}")
        
        print(f"GPU {gpu_id}: ✓ Completed! Processed {sample_count} samples in {batch_count + 1} batches")
        
    except Exception as e:
        print(f"GPU {gpu_id}: Error during processing: {e}")
        import traceback
        traceback.print_exc()

def save_batch_jsonl(batch, filepath):
    """Save a batch of new_anno formatted data as JSONL"""
    try:
        with open(filepath, 'w') as f:
            for sample in batch:
                f.write(json.dumps(sample) + '\n')
    except Exception as e:
        print(f"Error saving batch to {filepath}: {e}")

def main(args):
    """Main function to launch independent GPU processes"""
    world_size = torch.cuda.device_count()
    
    if world_size == 0:
        print("No GPUs available!")
        return
    
    print(f"Using {world_size} GPUs for independent tokenization")
    
    # Configuration
    config = {
        'folder_path': args.folder_path,
        'output_dir': args.output_dir,
        'batch_size': 512,
        'save_every': 100000,
        'num_workers': 8,
        'vocab_size': args.vocab_size
    }
    
    # Spawn independent processes - no distributed coordination needed
    mp.spawn(
        process_on_gpu,
        args=(world_size, config['folder_path'], config['output_dir'], 
              config['batch_size'], config['save_every'], config['num_workers'], config['vocab_size']),
        nprocs=world_size,
        join=True  # Wait for all to complete
    )
    
    print("All GPUs have completed processing!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()
    main(args)

# python tokenize_to_jsonl_laion_ibq.py --folder_path /path/to/data/filtered_laion_aesthetics_parquet_recaptioned/val.parquet --output_dir /path/to/data/laion_val_tokenized_ibq_1024 --vocab_size 1024