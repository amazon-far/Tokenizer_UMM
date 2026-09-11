
# --- path configuration (see .env.example) ---
import os
DATA_ROOT = os.environ.get("DATA_ROOT", "/path/to/data")

import torch
import  json
import os
import argparse
from torch.nn import functional as F
import sys
sys.path.append('../')
from PIL import Image
import yaml

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--model_path', type=str, default='/path/to/checkpoint')
    parser.add_argument('--save_path', type=str, default='GenAI_Bench_527_results')
    parser.add_argument('--benchmark_name', type=str, default='genai')
    parser.add_argument('--chunk_idx', type=int, default=0)
    return parser

def split_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def main(args):
    LLM_pth = args.model_path
    text_set_id = args.chunk_idx
    image_save_pth = '{}/'.format(args.save_path)

    fname = f"{int(text_set_id):03d}.json"
    out_path = os.path.join(image_save_pth, fname)
    # load the saved JSON list and populate `chunk` and `image_list`
    if not os.path.exists(out_path):
        raise FileNotFoundError(f"Saved file not found: {out_path}")

    with open(out_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        image_list = data['save_list']
        info_list = data['info_list']

    # ensure lists are plain Python lists (safe for subsequent processing)
    image_list = list(image_list)
    info_list = list(info_list)

    if args.benchmark_name == 'genai' or args.benchmark_name == 'dpg':
        suffix = '.jpg'
    else:
        suffix = '.png'

    if 'unitok' in LLM_pth:
        sys.path.append('../unitok/')
        from utils.config import Args
        from models.unitok import UniTok
        if 'unitok_sem' in LLM_pth:
            ckpt_path = DATA_ROOT + '/unitok_checkpoint_large_clip_1codebook_sem/ckpt-last.pth'
        else:
            ckpt_path = DATA_ROOT + '/unitok_checkpoint_large_clip_1codebook/ckpt-last.pth'
        ckpt = torch.load(ckpt_path, map_location='cpu')
        unitok_cfg = Args()
        unitok_cfg.load_state_dict(ckpt['args'])
        unitok = UniTok(unitok_cfg)
        unitok.load_state_dict(ckpt['trainer']['unitok'])
        unitok.to('cuda')
        unitok.eval()
        unitok.requires_grad_(False)
        for datainfo, vq_code in zip(info_list, image_list):
            if args.benchmark_name == "mjhq":
                name = datainfo['name']
                category = datainfo['category']
                sub_path = os.path.join(image_save_pth, category)
                if not os.path.exists(sub_path):
                    os.makedirs(sub_path)
                image_path = '{}/{}.png'.format(sub_path,name)
            else:
                idx = datainfo['Index']
                image_path = '{}/{}{}'.format(image_save_pth, str(idx), suffix)
            latents = torch.tensor(vq_code).to('cuda')
            reconstructed_image = unitok.idx_to_img(latents.unsqueeze(0).unsqueeze(0))
            reconstructed_image = reconstructed_image.add(1).mul_(0.5 * 255).round().nan_to_num_(128, 0, 255).clamp_(0, 255)
            reconstructed_image = reconstructed_image.to(dtype=torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
            Image.fromarray(reconstructed_image).save(image_path)
    elif 'gigatok' in LLM_pth:
        sys.path.append('../GigaTok/')
        from GigaTok.utils.model_init import load_model_from_config, custom_load
        model_config = "../GigaTok/configs/vq/VQ_BL256.yaml" if 'dino' not in LLM_pth else "../GigaTok/configs/vq/VQ_BL256_dino_disc.yaml"
        with open(model_config, "r") as f:
            config = yaml.safe_load(f)
        tokenizer_model = load_model_from_config(config)
        causal_type = config["model"]["causal_settings"]["causal_type"]
        print(causal_type) # None
        print(f"VQ Model Parameters(inference): {sum(p.numel() for p in tokenizer_model.parameters()):,}")
        ckpt_path = "../GigaTok/VQ_BL256_e200.pt" if 'dino' not in LLM_pth else "../GigaTok/VQ_BL256_dino_disc.pt"
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "ema" in checkpoint:  # ema
            model_weight = checkpoint["ema"]
        elif "model" in checkpoint:  # ddp
            model_weight = checkpoint["model"]
        elif "state_dict" in checkpoint:
            model_weight = checkpoint["state_dict"]
        else:
            raise Exception("please check model weight")

        # tokenizer_model.load_state_dict(model_weight)
        custom_load(tokenizer_model, model_weight)
        del checkpoint
        tokenizer_model.to("cuda")
        tokenizer_model.eval()
        tokenizer_model.requires_grad_(False)
        for datainfo, vq_code in zip(info_list, image_list):
            if args.benchmark_name == "mjhq":
                name = datainfo['name']
                category = datainfo['category']
                sub_path = os.path.join(image_save_pth, category)
                if not os.path.exists(sub_path):
                    os.makedirs(sub_path)
                image_path = '{}/{}.png'.format(sub_path,name)
            else:
                idx = datainfo['Index']
                image_path = '{}/{}{}'.format(image_save_pth, str(idx), suffix)
            vq_code = torch.tensor(vq_code).to('cuda')
            reconstructed_image = tokenizer_model.decode_code(vq_code, (1, 8, 1, 256)) # output value is between [-1, 1]
            reconstructed_image = torch.clamp(127.5 * reconstructed_image + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
            Image.fromarray(reconstructed_image).save(image_path)
    elif 'ibq' in LLM_pth:
        sys.path.append('../seed_voken/')
        from omegaconf import OmegaConf
        from src.IBQ.models.ibqgan import IBQ
        import numpy as np
        MODEL_TYPE = {
            "IBQ": IBQ
        }

        def load_vqgan_new(config, model_type, ckpt_path=None, is_gumbel=False):
            model = MODEL_TYPE[model_type](**config.model.init_args)
            if ckpt_path is not None:
                sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
                missing, unexpected = model.load_state_dict(sd, strict=False)
            return model.eval()
            
        if "ibq_8192" in LLM_pth:
            config_file = "../seed_voken/configs/IBQ/gpu/imagenet_ibqgan_8192.yaml"
            ckpt_path = "../seed_voken/imagenet256_8192.ckpt"
        elif "ibq_16384" in LLM_pth:    
            config_file = "../seed_voken/configs/IBQ/gpu/imagenet_ibqgan_16384.yaml"
            ckpt_path = "../seed_voken/imagenet256_16384.ckpt"
        elif "ibq_1024" in LLM_pth:
            config_file = "../seed_voken/configs/IBQ/gpu/imagenet_ibqgan_1024.yaml"
            ckpt_path = "../seed_voken/imagenet256_1024.ckpt"
        else:
            raise ValueError("Please check the IBQ checkpoint path!")
        model_name = "IBQ"
        configs = OmegaConf.load(config_file)
        ibq_tokenizer = load_vqgan_new(configs, model_name, ckpt_path)

        ibq_tokenizer.eval()
        ibq_tokenizer.requires_grad_(False)
        ibq_tokenizer = ibq_tokenizer.to(torch.device(f"cuda"))
        def custom_to_pil(x):
            x = x.detach().cpu()
            x = torch.clamp(x, -1., 1.)
            x = (x + 1.)/2.
            x = x.permute(1,2,0).numpy()
            x = (255*x).astype(np.uint8)
            x = Image.fromarray(x)
            if not x.mode == "RGB":
                x = x.convert("RGB")
            return x
        for datainfo, vq_code in zip(info_list, image_list):
            if args.benchmark_name == "mjhq":
                name = datainfo['name']
                category = datainfo['category']
                sub_path = os.path.join(image_save_pth, category)
                if not os.path.exists(sub_path):
                    os.makedirs(sub_path)
                image_path = '{}/{}.png'.format(sub_path,name)
            else:
                idx = datainfo['Index']
                image_path = '{}/{}{}'.format(image_save_pth, str(idx), suffix)
            latents = torch.tensor(vq_code).to('cuda')
            reconstructed_image = ibq_tokenizer.decode_code(latents, shape=(1, 16, 16, 256))
            reconstructed_image = custom_to_pil(reconstructed_image[0])
            reconstructed_image.save(image_path)
    else:
        sys.path.append('../chameleon/')
        from chameleon.inference.image_tokenizer import ImageTokenizer
        vqgan_cfg_path = "../../data/tokenizer/vqgan.yaml"
        vqgan_ckpt_path = "../../data/tokenizer/vqgan.ckpt"
        image_tokenizer = ImageTokenizer(  cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device="cuda",)
        
        if args.benchmark_name == 'dpg':
            for i in range(0, len(image_list), 4):
                datainfo = info_list[i]
                vq_codes = image_list[i:i+4]
                
                # Verify all datainfo entries in the group are the same
                for j in range(i + 1, i + 4):
                    if info_list[j] != datainfo:
                        raise ValueError(f"Different datainfo found at indices {i} and {j}")
                
                idx = datainfo['Index']
                image_path = '{}/{}{}'.format(image_save_pth, str(idx), suffix)
                
                imgs = [image_tokenizer.pil_from_img_toks(torch.tensor(code).to('cuda')) for code in vq_codes]
                w, h = imgs[0].size
                combined = Image.new('RGB', (w*2, h*2))
                combined.paste(imgs[0], (0, 0))
                combined.paste(imgs[1], (w, 0))
                combined.paste(imgs[2], (0, h))
                combined.paste(imgs[3], (w, h))
                combined.save(image_path)
        else:
            for datainfo, vq_code in zip(info_list, image_list):
                if args.benchmark_name == "mjhq":
                    name = datainfo['name']
                    category = datainfo['category']
                    sub_path = os.path.join(image_save_pth, category)
                    if not os.path.exists(sub_path):
                        os.makedirs(sub_path)
                    image_path = '{}/{}.png'.format(sub_path,name)
                else:
                    idx = datainfo['Index']
                    image_path = '{}/{}{}'.format(image_save_pth, str(idx), suffix)
                latents = torch.tensor(vq_code).to('cuda')
                rec_img = image_tokenizer.pil_from_img_toks(latents)
                rec_img.save(image_path)
        del image_tokenizer
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('image path check script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)