
# --- path configuration (see .env.example) ---
import os
DATA_ROOT = os.environ.get("DATA_ROOT", "/path/to/data")
HF_CACHE = os.environ.get("HF_CACHE", "/path/to/hf_cache")

import datasets 
import os
import glob
from datasets import load_dataset
from PIL import Image as PILImage
from tqdm import tqdm

data_files = glob.glob(os.path.join(DATA_ROOT + "/BLIP3o_60k", '*.tar')) 
dataset = load_dataset(
    "webdataset", 
    data_files=data_files, 
    cache_dir=HF_CACHE + '/', 
    split="train", 
    num_proc=64
)

# Create image directory
image_dir = DATA_ROOT + "/BLIP3o_60k_images"
os.makedirs(image_dir, exist_ok=True)

# Function to save image and return path
def save_image_and_get_path(example, idx):
    # Save image with a unique filename (using index or __key__)
    image_filename = f"{example['__key__']}.jpg"
    image_path = os.path.join(image_dir, image_filename)
    
    # Convert RGBA to RGB if necessary
    img = example['jpg']
    if img.mode == 'RGBA':
        # Create a white background
        rgb_img = PILImage.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        img = rgb_img
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Save the image
    img.save(image_path, 'JPEG')
    
    # Create the s3_image_url column (or use local path)
    example['s3_image_url'] = f"s3://your-s3-bucket/datasets/BLIP3o_60k_images/{image_filename}"
    
    # Remove the original jpg column to save space
    del example['jpg']
    
    return example

# Process dataset
dataset = dataset.map(save_image_and_get_path, with_indices=True, num_proc=16)

# Save to parquet
os.makedirs(DATA_ROOT + "/BLIP3o_60k_parquet", exist_ok=True)
dataset.to_parquet(DATA_ROOT + "/BLIP3o_60k_parquet/0.parquet")