
import os
from datasets import load_dataset
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import json
from PIL import Image, ImageFile
import torch
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
import argparse
import boto3
import io
from datetime import timedelta
from botocore.config import Config
from torchvision import transforms
from utils.config import Args
from models.unitok import UniTok
from utils.data import normalize_01_into_pm1
import yaml
import torch.nn.functional as F

BOTO3_CONFIG = Config(max_pool_connections=50, retries={'max_attempts': 5, 'mode': 'standard'})

ImageFile.LOAD_TRUNCATED_IMAGES = False
torch.set_grad_enabled(False)


def load_image_from_s3_uri(s3_uri, s3_client=None):
    """Load image from S3 URL"""
    # Example from JourneyDB: s3://your-s3-bucket/datasets/journeydb/ingest/assets/train/imgs/197/b8496271-7fb7-4859-8f4f-6d040dc8e3ff.jpg
    # Example from LAION-aesthetics: s3://your-s3-bucket/datasets/laion_aesthetics_v2_5.5/img2dataset/10_32/00030/000306419.jpg
    # image_path = s3_path.replace("s3://your-s3-bucket/datasets/journeydb/ingest/", "/path/to/data/predownloaded/datasets/Liquid/journeydb_ingest_all/")
    
    
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
    def __init__(self, streaming_dataset, tokenizer_config, rank=0):
        super().__init__()
        self.rank = rank
        self.img_size = tokenizer_config.img_size
        self.resize_ratio = tokenizer_config.resize_ratio
        print(f"Rank {rank}: Materializing dataset...")
        self.samples = []
        for sample in streaming_dataset:
            self.samples.append(sample)
        
        print(f"Rank {rank}: Dataset materialized with {len(self.samples)} samples")
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
            if sample['gpt3.5_caption'] is None:
                return None
            
            ori_image = load_image_from_s3_uri(sample['s3_uri'], self.s3_client)
            
            if ori_image is None:
                return None
            
            # Check aspect ratio
            original_ratio = max(ori_image.size) / min(ori_image.size)
            if original_ratio > 2:
                return None
            
            # Process image
            preprocess = transforms.Compose([
                transforms.Resize(int(self.img_size * self.resize_ratio)),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(), normalize_01_into_pm1,
            ])
            input_image = preprocess(ori_image)
            
            return {
                'image': input_image,
                'caption': sample['gpt3.5_caption'],
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


def setup_distributed(rank, world_size, port="12355"):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=600))
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def tokenize_batch(batch, image_tokenizer, rank=0):
    """Tokenize a batch of samples and format according to new_anno structure"""
    if batch is None or len(batch['captions']) == 0:
        return []
    try:
        images = batch['images'].to(torch.device(f"cuda:{rank}"))
        captions = batch['captions']
        
        with torch.no_grad():
            vqcodes = image_tokenizer.img_to_idx(images).cpu().view(len(captions), -1).tolist()
    
        results = []
        for idx, caption in enumerate(captions):
            new_anno = {
                "tokenized": True,
                "source": "{\"images\": \"midjourney\", \"captions\": \"gpt3.5\"}",
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
        print(f"Rank {rank}: Error processing batch: {e}")
        import traceback
        traceback.print_exc()
        return []

def load_and_tokenize_distributed(rank, world_size, folder_path, 
                                output_dir="tokenized_data",
                                batch_size=16,
                                save_every=500,
                                num_workers=4,
                                ckpt_path=None):
    """Main distributed processing function"""
    
    # Setup distributed processing
    setup_distributed(rank, world_size)
    
    try:
        print(f"Rank {rank}: Starting tokenization process")
        
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
        

        ckpt = torch.load(ckpt_path, map_location='cpu')
        unitok_cfg = Args()
        unitok_cfg.load_state_dict(ckpt['args'])
        unitok = UniTok(unitok_cfg)
        unitok.load_state_dict(ckpt['trainer']['unitok'])
        unitok = unitok.to(torch.device(f"cuda:{rank}"))
        unitok.eval()
        dataset = ImageTextDataset(streaming_dataset, unitok_cfg)
        # Create DataLoader with distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=1 if num_workers > 0 else None,
            worker_init_fn=worker_init_fn if num_workers > 0 else None
        )

        # Process and save data
        output_path = f"{output_dir}/rank_{rank}"
        os.makedirs(output_path, exist_ok=True)

        batch_count = 0
        sample_count = 0
        processed_samples = []
        # Process dataset in batches
        print(f"Rank {rank}: Starting to process samples...")
        for batch in tqdm(dataloader, desc=f"Rank {rank}"):
            if batch is None:
                continue
            tokenized_samples = tokenize_batch(batch, unitok, rank)
            processed_samples.extend(tokenized_samples)
            sample_count += len(tokenized_samples)
            
            # Save if we have enough samples
            if len(processed_samples) >= save_every:
                save_path = f"{output_path}/batch_{batch_count:06d}.jsonl"  # Changed to .jsonl
                save_batch_jsonl(processed_samples, save_path)
                
                print(f"Rank {rank}: Saved batch {batch_count}, total samples: {sample_count}")
                
                processed_samples = []
                batch_count += 1
            

        # Process remaining samples in processed_samples
        if processed_samples:
            save_path = f"{output_path}/batch_{batch_count:06d}.jsonl"
            save_batch_jsonl(processed_samples, save_path)
            print(f"Rank {rank}: Saved final batch {batch_count}")
        
        print(f"Rank {rank}: Completed processing {sample_count} samples in {batch_count + 1} batches")

        dist.barrier()
        
    except Exception as e:
        print(f"Rank {rank}: Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_distributed()

def save_batch_jsonl(batch, filepath):
    """Save a batch of new_anno formatted data as JSONL"""
    try:
        with open(filepath, 'w') as f:
            for sample in batch:
                f.write(json.dumps(sample) + '\n')
        print(f"Saved {len(batch)} samples to {filepath}")
    except Exception as e:
        print(f"Error saving batch to {filepath}: {e}")

def main(args):
    """Main function to launch distributed training"""
    world_size = torch.cuda.device_count()
    
    if world_size == 0:
        print("No GPUs available!")
        return
    
    print(f"Using {world_size} GPUs for distributed tokenization")
    
    # Configuration
    config = {
        'folder_path': args.folder_path,  # Path to parquet file or directory containing parquet files
        'output_dir': args.output_dir,
        'ckpt_path': args.ckpt_path,
        'batch_size': 512,  # Adjust based on GPU memory
        'save_every': 100000,  # Save every N samples
        'num_workers': 8
    }
    
    # Spawn processes
    mp.spawn(
        load_and_tokenize_distributed,
        args=(world_size, config['folder_path'], config['output_dir'],
              config['batch_size'], config['save_every'], config['num_workers'], config['ckpt_path']),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ckpt_path", required=True)
    args = parser.parse_args()
    main(args)

# python tokenize_to_jsonl_journeydb_unitok.py --ckpt_path '/path/to/data/unitok_checkpoint_large_clip_1codebook/ckpt-last.pth' \
# --folder_path /path/to/data/predownloaded/datasets/Liquid/journeydb_ingest_all/parquet/validation/ --output_dir /path/to/data/journeydb_val_tokenized_unitok