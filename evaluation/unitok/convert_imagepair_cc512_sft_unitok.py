import os
import json
import argparse
import PIL

from PIL import Image, ImageFile
import argparse
import torch
import torch.multiprocessing as mp
from torchvision import transforms
from utils.config import Args
from models.unitok import UniTok
from utils.data import normalize_01_into_pm1
import yaml
from tqdm import tqdm
import numpy as np
import datasets
from datasets import Dataset
from datasets import load_dataset
ImageFile.LOAD_TRUNCATED_IMAGES = False
torch.set_grad_enabled(False)


def process_item(batch,image_tokenizer,tokenizer_config,source,image_folder):
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
            
        
        
        # original_ratio = max(ori_image.size) / min(ori_image.size) # remove images with ratio>=3
        # if original_ratio >=3 :
        #     return None    
        try:
            ori_image = image.convert('RGB')
            # Process image
            preprocess = transforms.Compose([
                transforms.Resize(int(tokenizer_config.img_size * tokenizer_config.resize_ratio)),
                transforms.CenterCrop(tokenizer_config.img_size),
                transforms.ToTensor(), normalize_01_into_pm1,
            ])
            input_image = preprocess(ori_image)
        except Exception as e:
            print('crop error', e)
            continue

        input_images.append(input_image)
        new_batch.append(data)
            
    if len(input_images) > 0:
        input_images = torch.stack(input_images, dim=0).to(next(image_tokenizer.parameters()).device)
        with torch.no_grad():
            vqcode = image_tokenizer.img_to_idx(input_images).cpu().view(input_images.shape[0], -1).tolist()
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
            new_anno={  "tokenized": True,
                        "source":source,
                        'data_type': 'instruction',
                        'text': conversations,
                        'length': 0,
                        'vqcode_256': json.dumps(vqcode[counter]),
                        'vqcode_512': 'no',
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


def worker_process(rank, args, all_data, world_size):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    ckpt_path = args.ckpt_path
    ckpt = torch.load(ckpt_path, map_location='cpu')
    unitok_cfg = Args()
    unitok_cfg.load_state_dict(ckpt['args'])
    unitok = UniTok(unitok_cfg)
    unitok.load_state_dict(ckpt['trainer']['unitok'])
    unitok.to(device)
    unitok.eval()
    # Split data for this rank
    world_size = world_size
    rank_chunks = np.array_split(all_data, world_size)
    subset = rank_chunks[rank].tolist()
    print(f"GPU {rank}: Processing {len(subset)} items")

    # Use ThreadPoolExecutor instead of Pool for better GPU sharing
    valid_pair_list = []
    for start_index in tqdm(range(0, len(subset), args.batch_size), desc=f"GPU {rank} processing"):
        end_index = min(start_index + args.batch_size, len(subset))
        batch_results = process_item(subset[start_index:end_index], unitok, unitok_cfg, args.source_name, image_folder=args.image_path)
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
    if args.input_pairs.endswith('.jsonl') or args.input_pairs.endswith('.json'):
        with open(args.input_pairs, 'r') as f:
            all_data = json.load(f)
    else:
        try:
            all_data = load_dataset("parquet", data_dir=args.input_pairs)
        except:
            all_data = datasets.load_from_disk(args.input_pairs)
        all_data = all_data['train']
        all_data = all_data.to_list()

    print(len(all_data))
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for distributed processing")
    
    if world_size <= 1:
        print("Warning: Only 1 GPU available, falling back to single GPU processing")
        worker_process(0, args, all_data, world_size)
    else:
        # Use mp.spawn to launch processes on each GPU
        mp.spawn(worker_process, args=(args, all_data, world_size), nprocs=world_size, join=True)
    
    # Find all GPU result files
    gpu_files = []
    for i in range(world_size):
        gpu_file = os.path.join(args.temp_path, f'gpu_{i:03d}.jsonl')
        if os.path.exists(gpu_file):
            gpu_files.append(gpu_file)
    
    print(f"Found {len(gpu_files)} GPU result files")
    
    # Create dataset directly from all files
    if gpu_files:
        # ds = load_dataset('json', data_files=gpu_files, split='train')
        all_data = []
        for gpu_file in gpu_files:
            print(f"Sample data from {gpu_file}:")
            with open(gpu_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        item = json.loads(line)
                        # for i, sentence in enumerate(item['text']):
                        #     item['text'][i] = {'from': sentence['from'], 'value': sentence['value']}
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
    parser.add_argument('--ckpt_path', type=str, default='', help='path to the UniTok checkpoint')
    parser.add_argument('--input_pairs', type=str, default='/path/to/save/josnl',help='jsonl file, where image pairs meta saved')
    parser.add_argument('--image_path', type=str, default=None, help='if image path not in jsonl, set it here')
    parser.add_argument('--temp_path', type=str, default='/path/to/save/tempdata',help='path to save converted jsonl files')
    parser.add_argument('--save_path', type=str, default='/path/to/save/hfdata',help='path to save converted hf datasets files')
    parser.add_argument('--num_processes', type=int, default=8)
    parser.add_argument('--source_name', type=str)
    parser.add_argument('--vqgan_path', type=str, default='/path/to/vqgan_weights',help='where vqgan.yaml and vqgan.ckpt saved')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size for each process')
    
    args = parser.parse_args()
    main(args)

# python convert_imagepair_cc512_sft_unitok.py --ckpt_path '/path/to/data/unitok_checkpoint_large_clip_1codebook/ckpt-last.pth' --input_pairs /path/to/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json --image_path /path/to/data/LLaVA-Pretrain/images/ --temp_path /tmp/llava_pretrain_jsonl --save_path /path/to/data/llava_pretrain_tokenized_unitok  --source_name llava_pretrain 
# python convert_imagepair_cc512_sft_unitok.py --ckpt_path '/path/to/data/unitok_checkpoint_large_clip_1codebook/ckpt-last.pth' --input_pairs /path/to/data/playground/dvqa --temp_path /tmp/dvqa_jsonl --save_path /path/to/data/dvqa_tokenized_unitok  --source_name dvqa
# python convert_imagepair_cc512_sft_unitok.py --ckpt_path '/path/to/data/unitok_checkpoint_large_clip_1codebook/ckpt-last.pth' --input_pairs /path/to/data/playground/ai2d --temp_path /tmp/ai2d_jsonl --save_path /path/to/data/ai2d_tokenized_unitok  --source_name ai2d
# python convert_imagepair_cc512_sft_unitok.py --ckpt_path '/path/to/data/unitok_checkpoint_large_clip_1codebook/ckpt-last.pth' --input_pairs /path/to/data/playground/ALLaVA-4V/allava_laion/ALLaVA-Caption-LAION-4V.json --image_path /path/to/data/playground/ALLaVA-4V/  --temp_path /tmp/allava_laion_caption_jsonl --save_path /path/to/data/allava_laion_caption_tokenized_unitok  --source_name allava_laion_caption
# python convert_imagepair_cc512_sft_unitok.py --ckpt_path '/path/to/data/unitok_checkpoint_large_clip_1codebook/ckpt-last.pth' --input_pairs /path/to/data/playground/ALLaVA-4V/allava_vflan/ALLaVA-Caption-VFLAN-4V.json --image_path /path/to/data/playground/ALLaVA-4V/allava_vflan/  --temp_path /tmp/allava_vflan_caption_jsonl --save_path /path/to/data/allava_vflan_caption_tokenized_unitok  --source_name allava_vflan_caption
# python convert_imagepair_cc512_sft_unitok.py --ckpt_path '/path/to/data/unitok_checkpoint_large_clip_1codebook/ckpt-last.pth' --input_pairs /path/to/data/playground/mgm_instruction_noai2d_nodvqa.json --image_path /path/to/data/playground/  --temp_path /tmp/mgm_instruction_jsonl --save_path /path/to/data/mgm_instruction_tokenized_unitok  --source_name mgm_instruction