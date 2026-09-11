import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from utils.model_init import load_model_from_config, custom_load
from dataset.augmentation import center_crop_arr
import yaml

model_config = "configs/vq/VQ_BL256.yaml"
with open(model_config, "r") as f:
    config = yaml.safe_load(f)
tokenizer_model = load_model_from_config(config)
causal_type = config["model"]["causal_settings"]["causal_type"]
print(causal_type) # None
tokenizer_model.to("cuda")
tokenizer_model.eval()
print(f"VQ Model Parameters(inference): {sum(p.numel() for p in tokenizer_model.parameters()):,}")
ckpt_path = "VQ_BL256_e200.pt"

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

transform = transforms.Compose([
    transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])

image = Image.open("/path/to/Liquid/assets/v0.jpg").convert("RGB")

input_tensor = transform(image).contiguous().to('cuda')
input_tensor = torch.stack([input_tensor, input_tensor])
with torch.no_grad():
    latent, _, [_, _, indices] = tokenizer_model.encode(
                                    input_tensor, 
                                    num_en_q_level=None, 
                                    causal_type=causal_type)
    samples = tokenizer_model.decode_code(indices, latent.shape) # output value is between [-1, 1]
print(indices.shape)
print(latent.shape)
# torch.Size([256])
# torch.Size([1, 8, 1, 256])
samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
out_path = "/path/to/Liquid/assets/0_gigatok_bl256.png"
Image.fromarray(samples[0]).save(out_path)
print(f"Saved reconstructed image to {out_path}")