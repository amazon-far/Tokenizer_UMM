"""
Image Reconstruction code
"""
import os
import sys
sys.path.append(os.getcwd())
import PIL
import torch
from omegaconf import OmegaConf
import importlib
import numpy as np
from PIL import Image
from tqdm import tqdm
from src.Open_MAGVIT2.models.lfqgan import VQModel
from src.IBQ.models.ibqgan import IBQ
import argparse
try:
	import torch_npu
except:
    pass

if hasattr(torch, "npu"):
    DEVICE = torch.device("npu:0" if torch_npu.npu.is_available() else "cpu")
else:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## for different model configuration
MODEL_TYPE = {
    "Open-MAGVIT2": VQModel,
    "IBQ": IBQ
}

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

def load_vqgan_new(config, model_type, ckpt_path=None, is_gumbel=False):
	model = MODEL_TYPE[model_type](**config.model.init_args)
	if ckpt_path is not None:
		sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
		missing, unexpected = model.load_state_dict(sd, strict=False)
	return model.eval()

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

def main(args):
    config_file = args.config_file
    configs = OmegaConf.load(config_file)
    model = load_vqgan_new(configs, args.model, args.ckpt_path).to(DEVICE)
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    image_path = "/path/to/Liquid/assets/v7.jpg"
    out_path = image_path.replace('.jpg', '_{}.jpg'.format("ibq_16384"))
    out_path = out_path.replace('.jpeg', '_{}.jpeg'.format("ibq_16384"))
    out_path = out_path.replace('.png', '_{}.png'.format("ibq_16384"))
    with torch.no_grad():
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = center_crop_image(image, args.image_size, args.image_size)
        image = np.array(image).astype(np.uint8)
        image = (image/127.5 - 1.0).astype(np.float32)
        images = torch.from_numpy(image).unsqueeze(0).permute(0,3,1,2).to(DEVICE)
        # images = torch.stack([images.squeeze(0), images.squeeze(0)])
        if False: #model.use_ema:
            print("Using EMA for inference")
            with model.ema_scope():
                if args.model == "Open-MAGVIT2":
                    quant, diff, indices, _ = model.encode(images)
                elif args.model == "IBQ":
                    quant, qloss, (_, _, indices) = model.encode(images)
                reconstructed_images = model.decode(quant)
        else:
            if args.model == "Open-MAGVIT2":
                quant, diff, indices, _ = model.encode(images)
            elif args.model == "IBQ":
                quant, qloss, (_, _, indices) = model.encode(images)
            print(quant.shape) # torch.Size([1, 256, 16, 16])
            reconstructed_images = model.decode_code(indices, (quant.permute(0,2,3,1).shape))
        print(indices.shape) # torch.Size([256])
        reconstructed_image = reconstructed_images[0]
        reconstructed_image = custom_to_pil(reconstructed_image)

        reconstructed_image.save(out_path)
        original_image = custom_to_pil(images[0])
        original_image.save(image_path.replace('.jpg', '_original.jpg').replace('.jpeg', '_original.jpeg').replace('.png', '_original.png'))

    
def get_args():
   parser = argparse.ArgumentParser(description="inference parameters")
   parser.add_argument("--config_file", required=True, type=str)
   parser.add_argument("--ckpt_path", required=True, type=str)
   parser.add_argument("--image_size", default=256, type=int)
   parser.add_argument("--model", choices=["Open-MAGVIT2", "IBQ"])

   return parser.parse_args()
  
if __name__ == "__main__":
  args = get_args()
  main(args)

# python demo.py --config_file "configs/IBQ/gpu/imagenet_ibqgan_16384.yaml" --ckpt_path imagenet256_16384.ckpt --model IBQ 