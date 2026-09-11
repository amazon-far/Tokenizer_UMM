
# --- path configuration (see .env.example) ---
import os
DATA_ROOT = os.environ.get("DATA_ROOT", "/path/to/data")
CKPT_ROOT = os.environ.get("CKPT_ROOT", "/path/to/checkpoints")

import argparse
from email.mime import image
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from conversation import conv_templates
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile
from PIL import Image
import math
import sys
import PIL
sys.path.append('../')
import numpy as np
from torchvision import transforms
from typing import Any, cast

IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"

ImageFile.LOAD_TRUNCATED_IMAGES = False
torch.set_grad_enabled(False)


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

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

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, model_path=None):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.model_path = model_path

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        # import pdb;pdb.set_trace()
        
        # if self.model_config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompt = prompt.replace('<image>','<boi><image><eoi>')
        # import pdb;pdb.set_trace()
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        
        if 'unitok' in self.model_path.lower():
            sys.path.append('../unitok/')
            from utils.data import normalize_01_into_pm1
            preprocess = transforms.Compose([
                transforms.Resize(int(256 * 1.125)),
                transforms.CenterCrop(256),
                transforms.ToTensor(), normalize_01_into_pm1,
            ])
            img = preprocess(image)
            img = img.unsqueeze(0).to('cuda')
            with torch.no_grad():
                code_idx = self.image_processor.img_to_idx(img)
                vq_code = code_idx.view(-1)
        elif 'gigatok' in self.model_path.lower():
            from GigaTok.dataset.augmentation import center_crop_arr
            transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
            image_tensor = transform(image).unsqueeze(0).contiguous().to('cuda')
            with torch.no_grad():
                latent, _, [_, _, indices] = self.image_processor.encode(
                                        image_tensor, 
                                        num_en_q_level=None, 
                                        causal_type=None)
                vq_code = indices.view(-1)
        elif 'ibq' in self.model_path.lower():
            image = center_crop_image(image)
            image = np.array(image).astype(np.uint8)
            image = (image/127.5 - 1.0).astype(np.float32)
            image = torch.tensor(image)
            image = torch.einsum('hwc->chw', image)
            image = image.float().unsqueeze(0).to('cuda')
            with torch.no_grad():
                quant, qloss, (_, _, indices) = self.image_processor.encode(image)
                vq_code = indices.view(-1)
        else:
            pad_image = expand2square(image, (122, 116, 104) )
            input_image = pad_image.resize((512, 512), PIL.Image.LANCZOS)
            with torch.no_grad():
                vq_code =  self.image_processor.img_tokens_from_pil(input_image) 
        vqcode = vq_code.cpu() 
        vqcode = vqcode+ len(self.tokenizer)
 
        text_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        num_images = (text_ids == IMAGE_TOKEN_INDEX).sum()
        eoi =  torch.tensor([8])
        boi =  torch.tensor([7])
        image_token_indices = [-1] + torch.where(text_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [text_ids.shape[0]]
        cur_input_ids = []
        for i in range(num_images + 1):
            cur_input_ids.append(text_ids[image_token_indices[i]+1:image_token_indices[i+1]])
            if i < num_images:
                input_vqcodes = torch.cat( [boi,vqcode,eoi],dim=0 )
                cur_input_ids.append( input_vqcodes )
        input_ids = torch.cat(cur_input_ids, dim=0)
        attention_mask = torch.ones_like(input_ids)
        return input_ids,  attention_mask, os.path.join(self.image_folder, image_file) #, image_tensor, image_tensor_aux
    
    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, model_path=None, batch_size=1, num_workers=0):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, model_path=model_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, load_8bit=args.load_8bit)

    
    kwargs = {}
    kwargs = {"device_map": "cuda"}
    if args.load_8bit:
        kwargs['load_in_8bit'] = True
    else:
        kwargs['torch_dtype'] = torch.float16

    if args.use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    # import pdb;pdb.set_trace()

    if 'unitok' in args.model_path.lower():
        sys.path.append('../unitok/')
        from utils.config import Args
        from models.unitok import UniTok
        if 'unitok_sem' in args.model_path.lower():
            ckpt_path = DATA_ROOT + '/unitok_checkpoint_large_clip_1codebook_sem/ckpt-last.pth'
        else:
            ckpt_path = DATA_ROOT + '/unitok_checkpoint_large_clip_1codebook/ckpt-last.pth'
        ckpt = torch.load(ckpt_path, map_location='cpu')
        unitok_cfg = Args()
        unitok_cfg.load_state_dict(ckpt['args'])
        image_tokenizer = UniTok(unitok_cfg)
        image_tokenizer.load_state_dict(ckpt['trainer']['unitok'])
        image_tokenizer.to('cuda')
        image_tokenizer.eval()
        image_tokenizer.requires_grad_(False)
    elif 'gigatok' in args.model_path.lower():
        sys.path.append('../GigaTok/')
        import yaml
        from GigaTok.utils.model_init import load_model_from_config, custom_load
        model_config = "../GigaTok/configs/vq/VQ_BL256.yaml" if 'dino' not in args.model_path.lower() else "../GigaTok/configs/vq/VQ_BL256_dino_disc.yaml"
        with open(model_config, "r") as f:
            config = yaml.safe_load(f)
        image_tokenizer = load_model_from_config(config)
        causal_type = config["model"]["causal_settings"]["causal_type"]
        print(causal_type) # None
        print(f"VQ Model Parameters(inference): {sum(p.numel() for p in image_tokenizer.parameters()):,}")
        ckpt_path = "../GigaTok/VQ_BL256_e200.pt" if 'dino' not in args.model_path.lower() else "../GigaTok/VQ_BL256_dino_disc.pt"
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
        image_tokenizer.to(torch.device(f"cuda"))
        image_tokenizer.eval()
        image_tokenizer.requires_grad_(False)
    elif 'ibq' in args.model_path.lower():
        sys.path.append('../seed_voken/')
        from omegaconf import OmegaConf
        # from src.Open_MAGVIT2.models.lfqgan import VQModel
        from src.IBQ.models.ibqgan import IBQ
        MODEL_TYPE = {
            "IBQ": IBQ
        }

        def load_vqgan_new(config, model_type, ckpt_path=None, is_gumbel=False):
            model = MODEL_TYPE[model_type](**config.model.init_args)
            if ckpt_path is not None:
                sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
                missing, unexpected = model.load_state_dict(sd, strict=False)
            return model.eval()
        if "ibq_8192" in args.model_path.lower():
            config_file = "../seed_voken/configs/IBQ/gpu/imagenet_ibqgan_8192.yaml"
            ckpt_path = "../seed_voken/imagenet256_8192.ckpt"
        elif "ibq_16384" in args.model_path.lower():    
            config_file = "../seed_voken/configs/IBQ/gpu/imagenet_ibqgan_16384.yaml"
            ckpt_path = "../seed_voken/imagenet256_16384.ckpt"
        elif "ibq_1024" in args.model_path.lower():
            config_file = "../seed_voken/configs/IBQ/gpu/imagenet_ibqgan_1024.yaml"
            ckpt_path = "../seed_voken/imagenet256_1024.ckpt"
        else:
            raise ValueError("Please check the IBQ checkpoint path!")
        model_name = "IBQ"
        configs = OmegaConf.load(config_file)
        image_tokenizer = load_vqgan_new(configs, model_name, ckpt_path)

        image_tokenizer.eval()
        image_tokenizer.requires_grad_(False)
        image_tokenizer = image_tokenizer.to(torch.device(f"cuda"))
    else:
        from chameleon.inference.image_tokenizer import ImageTokenizer
        vqgan_cfg_path = "../../data/tokenizer/vqgan.yaml"
        vqgan_ckpt_path = "../../data/tokenizer/vqgan.ckpt"
        image_tokenizer = ImageTokenizer(  cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device="cuda",)

    image_processor = image_tokenizer
    # import pdb;pdb.set_trace()

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, args.model_path)

    for (input_ids , attention_mask, imagepath), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]
        
        input_ids = input_ids.to(device=model.device, non_blocking=True)
        attention_mask = attention_mask.to(device=model.device, non_blocking=True)
        if hasattr(model, "update_prompt"):
            model.update_prompt([[cur_prompt]])
        with torch.inference_mode():
            
            inputs_embeds = model.model.embed_tokens(input_ids)
            output_ids = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                bos_token_id=tokenizer.bos_token_id,  # Begin of sequence token
                eos_token_id=tokenizer.eos_token_id,  # End of sequence token
                pad_token_id=tokenizer.pad_token_id,  # Pad token
                use_cache=False)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # print('imagepath\n',imagepath)
        # print('question\n',cur_prompt)
        # print('answer:\n', outputs)
        # import pdb;pdb.set_trace()  # cur_prompt,outputs
        
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument('--use_flash_attn', type=bool, default=True)
    parser.add_argument('--load_8bit', type=bool, default=False)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--image-resolution", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
