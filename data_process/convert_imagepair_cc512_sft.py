
# --- path configuration (see .env.example) ---
import os
HF_CACHE = os.environ.get("HF_CACHE", "/path/to/hf_cache")

import os
import json
import argparse
import PIL

from PIL import Image, ImageFile
import argparse
import torch
import torch.multiprocessing as mp
from vqgan.image_tokenizer import ImageTokenizer
from tqdm import tqdm
import numpy as np
import datasets
from datasets import Dataset
from datasets import load_dataset
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

def process_item_sat(data,image_tokenizer):
    image_paths = data['images']
    images = [Image.open(os.path.join("/your-path", image_file)) for image_file in image_paths]
    if len(images) == 2:
        image = Image.new('RGB', (images[0].width + images[1].width, max(images[0].height, images[1].height)))
        image.paste(images[0], (0, 0))
        image.paste(images[1], (images[0].width, 0))
    elif len(images) == 1:
        image = images[0]
    # try:
    #     ori_image = Image.open(image_path)
    # except:
    #     print('load error', image_path)
    #     return None
    ori_image = image
    original_ratio = max(ori_image.size) / min(ori_image.size) # remove images with ratio>=3
    if original_ratio >=3 :
        return None    

    input_image = center_crop_image(ori_image)

    with torch.no_grad():
        vqcode = image_tokenizer.img_tokens_from_pil(input_image) 
        vqcode = vqcode.cpu().tolist()

    new_anno={    
                'data_type': 'instruction',
                'text': data['messages'],
                'length': 0,
                'vqcode_256': 'no',
                'vqcode_512': json.dumps(vqcode),
                'vqcode_multi768': 'no',
                'width': 'no',
                'height': 'no',
                }
    return  new_anno

def process_item(batch,image_tokenizer,source,image_folder):
    input_images = []
    new_batch = []
    for data in batch:
        if 'image' in data:
            image_data = data['image']
        elif 'images' in data:
            image_data = data['images'][0]
        else:
            new_batch.append(data) 
            continue

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
            
        ori_image = image
        # original_ratio = max(ori_image.size) / min(ori_image.size) # remove images with ratio>=3
        # if original_ratio >=3 :
        #     return None    
        try:
            input_image = center_crop_image(ori_image)
        except Exception as e:
            print('crop error', e)
            continue
        
        input_images.append(input_image)
        new_batch.append(data)
            
    if len(input_images) > 0:
        with torch.no_grad():
            vqcode = image_tokenizer.img_tokens_from_pil(input_images) 
            vqcode = vqcode.cpu().view(len(input_images), -1).tolist()

    results = []
    counter = 0
    for data in new_batch:
        if 'conversations' in data:
            conversations = []
            for item in data['conversations']:
                conversations.append(
                    {
                        'from': item['from'],
                        'value': item['value']
                    }
                )
        elif 'texts' in data:
            conversations = []
            for item in data['texts']:
                conversations.append(
                    {
                        'from': 'human',
                        'value': item['user']
                    }
                )
                conversations.append(
                    {
                        'from': 'gpt',
                        'value': item['assistant']
                    }
                )
        if 'image' in data or 'images' in data:
            source_name = source
            new_anno={  "tokenized": True,
                        "source":source_name,
                        'data_type': 'instruction',
                        'text': conversations,
                        'length': 0,
                        'vqcode_256': 'no',
                        'vqcode_512': json.dumps(vqcode[counter]),
                        'vqcode_multi768': 'no',
                        'width': 'no',
                        'height': 'no',
                        }
            counter += 1
        else:
            new_anno={  "tokenized": True,
                        "source":source,
                        'data_type': 'instruction',
                        'text': conversations,
                        'length': 0,
                        'vqcode_256': 'no',
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
    vqgan_cfg_path = "{}/vqgan.yaml".format(args.vqgan_path)
    vqgan_ckpt_path = "{}/vqgan.ckpt".format(args.vqgan_path)
    image_tokenizer = ImageTokenizer(cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device=device)
    
    # Load dataset in this worker
    if dataset_path.endswith('.jsonl') or dataset_path.endswith('.json'):
        with open(dataset_path, 'r') as f:
            all_data = json.load(f)
        # Split for this rank
        rank_chunks = np.array_split(all_data, world_size)
        subset = rank_chunks[rank].tolist()
    elif dataset_path.endswith('.parquet'):
        dataset = load_dataset("parquet", data_files=dataset_path, cache_dir=HF_CACHE + '/', num_proc=32)
        if 'train' in dataset:
            dataset = dataset['train']
        subset = dataset.shard(num_shards=world_size, index=rank)
        print(f"GPU {rank}: Processing {len(subset)} items")
    else:
        # Load and shard the dataset for this GPU
        try:
            dataset = load_dataset("parquet", data_dir=dataset_path, cache_dir=HF_CACHE + '/', num_proc=32)
        except:
            dataset = datasets.load_from_disk(dataset_path)
        
        if 'train' in dataset:
            dataset = dataset['train']
        
        # Shard for this GPU - this is memory efficient!
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
        
        batch_results = process_item(batch, image_tokenizer, args.source_name, image_folder=args.image_path)
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
    parser.add_argument('--input_pairs', type=str, default='/path/to/save/josnl',help='jsonl file, where image pairs meta saved')
    parser.add_argument('--image_path', type=str, default=None, help='if image path not in jsonl, set it here')
    parser.add_argument('--temp_path', type=str, default='/path/to/save/tempdata',help='path to save converted jsonl files')
    parser.add_argument('--save_path', type=str, default='/path/to/save/hfdata',help='path to save converted hf datasets files')
    parser.add_argument('--source_name', type=str)
    parser.add_argument('--vqgan_path', type=str, default='/path/to/vqgan_weights',help='where vqgan.yaml and vqgan.ckpt saved')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for each process')
    
    args = parser.parse_args()
    main(args)

# python convert_imagepair_cc512_sft.py --input_pairs /path/to/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json --image_path /path/to/data/LLaVA-Pretrain/images/ --temp_path /tmp/llava_pretrain_jsonl --save_path /path/to/data/llava_pretrain_tokenized --vqgan_path /path/to/Tokenizer_UMM/data/tokenizer --source_name llava_pretrain 
# python convert_imagepair_cc512_sft.py --input_pairs /path/to/data/playground/dvqa --temp_path /tmp/dvqa_jsonl --save_path /path/to/data/dvqa_tokenized --vqgan_path /path/to/Tokenizer_UMM/data/tokenizer --source_name dvqa
# python convert_imagepair_cc512_sft.py --input_pairs /path/to/data/playground/ai2d --temp_path /tmp/ai2d_jsonl --save_path /path/to/data/ai2d_tokenized --vqgan_path /path/to/Tokenizer_UMM/data/tokenizer --source_name ai2d
# python convert_imagepair_cc512_sft.py --input_pairs /path/to/data/playground/ALLaVA-4V/allava_laion/ALLaVA-Caption-LAION-4V.json --image_path /path/to/data/playground/ALLaVA-4V/  --temp_path /tmp/allava_laion_caption_jsonl --save_path /path/to/data/allava_laion_caption_tokenized --vqgan_path /path/to/Tokenizer_UMM/data/tokenizer --source_name allava_laion_caption
# python convert_imagepair_cc512_sft.py --input_pairs /path/to/data/playground/ALLaVA-4V/allava_vflan/ALLaVA-Caption-VFLAN-4V.json --image_path /path/to/data/playground/ALLaVA-4V/allava_vflan/  --temp_path /tmp/allava_vflan_caption_jsonl --save_path /path/to/data/allava_vflan_caption_tokenized --vqgan_path /path/to/Tokenizer_UMM/data/tokenizer --source_name allava_vflan_caption
# python convert_imagepair_cc512_sft.py --input_pairs /path/to/data/playground/mgm_instruction_noai2d_nodvqa.json --image_path /path/to/data/playground/  --temp_path /tmp/mgm_instruction_jsonl --save_path /path/to/data/mgm_instruction_tokenized --vqgan_path /path/to/Tokenizer_UMM/data/tokenizer --source_name mgm_instruction
# python convert_imagepair_cc512_sft.py --input_pairs /path/to/data/playground/lmsys_chat_1m.json --image_path /path/to/data/playground/  --temp_path /tmp/lmsys_instruction_jsonl --save_path /path/to/data/lmsys_instruction_tokenized --vqgan_path /path/to/Tokenizer_UMM/data/tokenizer --source_name lmsys_instruction