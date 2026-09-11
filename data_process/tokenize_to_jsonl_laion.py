

# --- path configuration (see .env.example) ---
import os
REPO_ROOT = os.environ.get("REPO_ROOT", "/path/to/Tokenizer_UMM")

import os
import subprocess
from datasets import load_dataset
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import PIL
import json
from PIL import Image, ImageFile
import torch
import io
import boto3
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
from vqgan.image_tokenizer import ImageTokenizer
import numpy as np
from PIL import Image
import argparse
from datetime import timedelta
from botocore.config import Config

BOTO3_CONFIG = Config(max_pool_connections=50, retries={'max_attempts': 5, 'mode': 'standard'})

ImageFile.LOAD_TRUNCATED_IMAGES = False
torch.set_grad_enabled(False)

def center_crop_image(ori_image, tgt_width=512, tgt_height=512):
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

def whiten_transparency(img: PIL.Image) -> PIL.Image:
    # Check if it's already in RGB format.
    if img.mode == "RGB":
        return img

    vals_rgba = np.array(img.convert("RGBA"))

    # If there is no transparency layer, simple convert and return.
    if not (vals_rgba[:, :, 3] < 255).any():
        return img.convert("RGB")

    # There is a transparency layer, blend it with a white background.

    # Calculate the alpha proportion for blending.
    alpha = vals_rgba[:, :, 3] / 255.0
    # Blend with white background.
    vals_rgb = (1 - alpha[:, :, np.newaxis]) * 255 + alpha[:, :, np.newaxis] * vals_rgba[:, :, :3]
    return PIL.Image.fromarray(vals_rgb.astype("uint8"), "RGB")

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
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        print(f"Error loading image from {s3_uri}: {e}")
        return None


class ImageTextDataset(Dataset):
    """Dataset wrapper for streaming datasets"""
    def __init__(self, streaming_dataset, rank=0):
        super().__init__()
        self.rank = rank
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
            img = center_crop_image(ori_image)
            img = whiten_transparency(img)
            # vqgan_input = self._vqgan_input_from(image).to(self._device).to(self._dtype)
            # to support any resolution
            np_img = np.array(img) / 255.0  # Normalize to [0, 1]
            np_img = np_img * 2 - 1  # Scale to [-1, 1]
            img = torch.from_numpy(np_img).permute(2, 0, 1)
            img = img.float()
            
            return {
                'image': img,
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

def tokenize_batch(batch, encoder, rank=0):
    """Tokenize a batch of samples and format according to new_anno structure"""
    if batch is None or len(batch['captions']) == 0:
        return []
    try:
        images = batch['images'].to(torch.device(f"cuda:{rank}"))
        captions = batch['captions']
        
        with torch.no_grad():
            _, _, [_, _, img_toks] = encoder._vq_model.encode(images)
            vqcodes = img_toks.cpu().view(len(captions), -1).tolist()
                
        results = []
        for idx, caption in enumerate(captions):
            new_anno = {
                "tokenized": True,
                "source": "{\"images\": \"laion-aesthetics\", \"captions\": \"internvl3_1b\"}",   # NOTE: which caption hsould we use?
                'data_type': 'image_text',
                'text': caption,
                'length': len(caption) + len(vqcodes[idx]) * 4,
                'vqcode_256': 'no',
                'vqcode_512': json.dumps(vqcodes[idx]),
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
                                num_workers=4):
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
        
        dataset = ImageTextDataset(streaming_dataset)
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
        model_path = REPO_ROOT + "/data/tokenizer/vqgan.ckpt"
        config_path = REPO_ROOT + "/data/tokenizer/vqgan.yaml"
        encoder = ImageTokenizer(  cfg_path=config_path, ckpt_path=model_path, device=torch.device(f"cuda:{rank}"),)
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
            tokenized_samples = tokenize_batch(batch, encoder, rank)
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
        'batch_size': 128,  # Adjust based on GPU memory
        'save_every': 100000,  # Save every N samples
        'num_workers': 8
    }
    
    # Spawn processes
    mp.spawn(
        load_and_tokenize_distributed,
        args=(world_size, config['folder_path'], config['output_dir'], 
              config['batch_size'], config['save_every'], config['num_workers']),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    main(args)

# python tokenize_to_jsonl_laion.py --folder_path /path/to/data/filtered_laion_aesthetics_parquet/part_8.parquet --output_dir /path/to/data/laion_part_8_tokenized
# python tokenize_to_jsonl_laion.py --folder_path /path/to/data/filtered_laion_aesthetics_parquet/val.parquet --output_dir /path/to/data/laion_val_tokenized