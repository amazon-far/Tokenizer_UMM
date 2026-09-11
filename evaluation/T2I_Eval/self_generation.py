import transformers
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, GemmaForCausalLM
from torch import nn
import torch
import  json
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm
import argparse
from torch.nn import functional as F
import sys
sys.path.append('../')

from chameleon.inference.image_tokenizer import ImageTokenizer
import  numpy as np
from transformers import BitsAndBytesConfig

### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
    ):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """

    logits[:,:256000]=filter_value # sampling on VQVAE vocabulary for image generation
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample(logits, valid_min=None, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True):        
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    if valid_min is not None:
        mask = torch.arange(logits.size(-1), device=logits.device) < valid_min
        logits = logits.masked_fill(mask, float('-inf'))
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--model_path', type=str, default='/path/to/checkpoint')
    parser.add_argument('--save_path', type=str, default='GenAI_Bench_527_results')
    parser.add_argument('--load_8bit', type=bool, default=False)
    parser.add_argument('--chunk_idx', type=int, default=0)
    parser.add_argument('--num_chunks', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--cfg_scale', type=float, default=7.0)
    parser.add_argument('--tau', type=float, default=0.99)
    parser.add_argument('--topk', type=int, default=4096)
    parser.add_argument('--topp', type=float, default=0.96)
    parser.add_argument('--image_length', type=int)
    parser.add_argument('--max_image_token_id', type=int)
    return parser

def split_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def main(args):
    LLM_pth = args.model_path
    text_set_id = args.chunk_idx
    cfg_scale = args.cfg_scale
    use_catch = True
    tau = args.tau
    topk = args.topk
    topp = args.topp
    num_chunks=args.num_chunks
    batch_size = args.batch_size
    # image_save_pth = '{}/CFG{}_topk{}_topp{}_tau_{}'.format(args.save_path,str(cfg_scale), str(topk),str(topp),str(tau))
    image_save_pth = '{}/'.format(args.save_path)
    tokenizer = AutoTokenizer.from_pretrained(LLM_pth,padding_side='left')
    # vqllm = AutoModelForCausalLM.from_pretrained(
    #     LLM_pth,
    #     attn_implementation='flash_attention_2',
    #     torch_dtype=torch.bfloat16,
    #     load_in_8bit=args.load_8bit,
    #     )
    # if not args.load_8bit:
    #     vqllm = vqllm.to('cuda')
    if args.load_8bit:
        print("Load 8bit")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        
        vqllm = AutoModelForCausalLM.from_pretrained(
            LLM_pth,
            attn_implementation='flash_attention_2',
            quantization_config=quantization_config,
            device_map="auto",  # Required when using quantization
        )
    else:
        vqllm = AutoModelForCausalLM.from_pretrained(
            LLM_pth,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            device_map="auto",  # Or use .to('cuda') after loading
        )
    ori_vocabe_size = len(tokenizer)
    with open('./myprompts.txt', 'r') as f:
        lines = f.readlines()
    all_prompts = []
    for index,line in enumerate(lines):
        all_prompts.append({ 'Index':str(index+1).zfill(5),  'Prompt':line.strip()}) 

    chunked_filenames = np.array_split(all_prompts, num_chunks)
    subset = chunked_filenames[text_set_id].tolist()
    chunk_inputs = split_list(subset, batch_size)
    save_list = []
    info_list = []

    for chunk in chunk_inputs:

        text_inputs = [v['Prompt'] for v in chunk]
        uncondition_text_inputs = ['<unconditional><boi>']*len(text_inputs)
        for i in range(len(text_inputs)):
            text_inputs[i] = text_inputs[i]+' Generate an image based on this description.<boi>'
        if cfg_scale>1:
            model_inputs = tokenizer(text_inputs+uncondition_text_inputs, return_tensors="pt",padding=True).to('cuda')
        else:
            model_inputs = tokenizer(text_inputs, return_tensors="pt",padding=True).to('cuda')
        with torch.no_grad():
            sampling_kwargs={'temperature': tau, 'top_k': topk, 'top_p': topp, 'sample_logits': True}
            input_ids = model_inputs['input_ids']
            cur_len = input_ids.shape[1]
            model_kwargs = {'attention_mask':model_inputs['attention_mask']  , 'use_cache': True}
            model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

            pred_tokens = []
            for i in range(args.image_length):
                model_inputs = vqllm.prepare_inputs_for_generation(input_ids, **model_kwargs)
                if i > 0 and cfg_scale>1:
                    outputs = vqllm(
                        **model_inputs,
                        return_dict=True,
                        output_attentions=False,
                        output_hidden_states=False,
                    )
                else:
                    outputs = vqllm(
                        **model_inputs,
                        return_dict=True,
                        output_attentions=False,
                        output_hidden_states=False,
                    )

                next_token_logits = outputs.logits[:, -1:, :]
                
                if cfg_scale>1:
                    cond_logits, uncond_logits = torch.split(next_token_logits, len(next_token_logits) // 2, dim=0) 
                    cfg_logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
                    half_next_token, _ = sample(cfg_logits, valid_min=ori_vocabe_size, **sampling_kwargs)
                    pred_tokens.append(half_next_token)
                    next_token = torch.cat([half_next_token,half_next_token])

                else:
                    next_token, next_prob = sample(next_token_logits, valid_min=ori_vocabe_size, **sampling_kwargs)
                    pred_tokens.append(next_token)

                # update generated ids, model inputs, and length for next step
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                model_kwargs = vqllm._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=vqllm.config.is_encoder_decoder,
                )

            del sampling_kwargs
            del model_inputs
            del outputs
            image_vq_id = torch.cat(pred_tokens,dim=1)-ori_vocabe_size

            text = tokenizer.batch_decode(image_vq_id+ori_vocabe_size, skip_special_tokens=True)
            # save decoded text for this batch to a json file
            text_output = {
                "batch_indices": [v['Index'] for v in chunk],
                "token_ids": image_vq_id.cpu().numpy().tolist(),
                "decoded_texts": text
            }
            text_file = os.path.join(image_save_pth, f"{int(text_set_id):03d}_{chunk[0]['Index']}_text.json")
            with open(text_file, 'w', encoding='utf-8') as tf:
                json.dump(text_output, tf, ensure_ascii=False, indent=2)

            image_vq_id = torch.clamp(image_vq_id, min=0, max=args.max_image_token_id)
            image_list = []
            for index, generate_id in enumerate(image_vq_id):
                image_list.append(generate_id.tolist())
            save_list.extend(image_list)
            for info in chunk:
                info_list.append({'Index': info['Index']})
        torch.cuda.empty_cache()

    if not os.path.exists(image_save_pth):
        os.makedirs(image_save_pth)
    fname = f"{int(text_set_id):03d}.json"
    out_path = os.path.join(image_save_pth, fname)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({"save_list": save_list, "info_list": info_list}, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('image path check script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)