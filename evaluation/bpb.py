
# --- path configuration (see .env.example) ---
import os
REPO_ROOT = os.environ.get("REPO_ROOT", "/path/to/Tokenizer_UMM")
DATA_ROOT = os.environ.get("DATA_ROOT", "/path/to/data")
CKPT_ROOT = os.environ.get("CKPT_ROOT", "/path/to/checkpoints")

import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from tqdm import tqdm
from transformers import BitsAndBytesConfig

import os
import json

from liquid.constants import IGNORE_INDEX
import argparse

def setup_distributed():
    """Initialize distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def evaluate_bpb(model_name, LLM_pth, dataset_paths, text_field='text', split='train', 
                 max_length=2048, num_samples=None, load_8bit=False, vq_resolution=256, T2I_ratio=0.8):
    """
    Evaluate a Hugging Face language model using bits per byte metric on Arrow format dataset.
    
    Args:
        LLM_pth: HF model name or path
        dataset_path: Path to Arrow dataset directory or HF dataset name
        text_field: Field name containing text in dataset
        split: Dataset split to use (e.g., 'train', 'test', 'validation')
        max_length: Maximum sequence length for evaluation
        num_samples: Limit number of samples (None = use all)
        streaming: Whether to use streaming mode for large datasets
    
    Returns:
        dict with bits_per_byte and other metrics
    """
    # Load model and tokenizer
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0
    
    # Load model and tokenizer
    if is_main_process:
        print(f"Loading model: {LLM_pth}")
        print(f"World size: {world_size}, Rank: {rank}")
        
    tokenizer = AutoTokenizer.from_pretrained(LLM_pth,padding_side='left')
    
    if load_8bit:
        if is_main_process:
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
            # device_map={'': local_rank} if world_size > 1 else "auto",
        )
    else:
        vqllm = AutoModelForCausalLM.from_pretrained(
            LLM_pth,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            # device_map={'': local_rank} if world_size > 1 else "auto"
        )
    if world_size > 1:
        vqllm = vqllm.to(f'cuda:{local_rank}')
    else:
        vqllm = vqllm.to('cuda')
    ori_vocab_size = len(tokenizer)
    vqllm.eval()

    # Prepare results dict (JSON-serializable)
    id2name = {0: "Text", 1: "JourneyDB_T2I", 2: "JourneyDB_Caption", 3: "JourneyDB_Image", 4: "JourneyDB_Text", 
    5: "LAION_T2I", 6: "LAION_Caption", 7: "LAION_Image", 8: "LAION_Text", 
    9: "BLIP3o_T2I", 10: "BLIP3o_Caption", 11: "BLIP3o_Image", 12: "BLIP3o_Text", 
    13: "GenAI-Bench-527_T2I", 14: "GenAI-Bench-527_Caption", 15: "GenAI-Bench-527_Image", 16: "GenAI-Bench-527_Text", 
    17: "GenAI-Bench-527_T2I_basic", 18: "GenAI-Bench-527_T2I_advanced", 
    19: "GenAI-Bench-527_T2I_attribute", 20: "GenAI-Bench-527_T2I_scene", 21: "GenAI-Bench-527_T2I_spatial_relation", 22: "GenAI-Bench-527_T2I_action_relation", 23: "GenAI-Bench-527_T2I_part_relation",
    24: "GenAI-Bench-527_T2I_counting", 25: "GenAI-Bench-527_T2I_comparison", 26: "GenAI-Bench-527_T2I_differentiation", 27: "GenAI-Bench-527_T2I_negation", 28: "GenAI-Bench-527_T2I_universal",
    29: "GenAI-Bench-527_Caption_basic", 30: "GenAI-Bench-527_Caption_advanced", 
    31: "GenAI-Bench-527_Caption_attribute", 32: "GenAI-Bench-527_Caption_scene", 33: "GenAI-Bench-527_Caption_spatial_relation", 34: "GenAI-Bench-527_Caption_action_relation", 35: "GenAI-Bench-527_Caption_part_relation",
    36: "GenAI-Bench-527_Caption_counting", 37: "GenAI-Bench-527_Caption_comparison", 38: "GenAI-Bench-527_Caption_differentiation", 39: "GenAI-Bench-527_Caption_negation", 40: "GenAI-Bench-527_Caption_universal",
    41: "LAION_T2I_force", 42: "BLIP3o_T2I_force", 43: "JourneyDB_T2I_force", 44: "GenAI-Bench-527_T2I_force",
    45: "LAION_Image_force", 46: "BLIP3o_Image_force", 47: "JourneyDB_Image_force", 
    48: "Text_force", 49: "JourneyDB_Caption_force", 50: "LAION_Caption_force", 51: "BLIP3o_Caption_force", 52: "GenAI-Bench-527_Caption_force",
    53: "JourneyDB_Text_force", 54: "LAION_Text_force", 55: "BLIP3o_Text_force", 56: "GenAI-Bench-527_Text_force"
    }
    name2id = {v: k for k, v in id2name.items()}
    with open(REPO_ROOT + '/evaluation/T2I_Eval/prompt_to_skills.json', 'r', encoding='utf-8') as f:
        prompt_to_skills = json.load(f)
    
    datasets_num = id2name.__len__() 
    total_nll = [0.0 for _ in range(datasets_num)]
    total_bytes = [0 for _ in range(datasets_num)]
    total_tokens = [0 for _ in range(datasets_num)]
    num_processed = [0 for _ in range(datasets_num)]
    # Load dataset
    for idx, dataset_path in enumerate(dataset_paths):
        if is_main_process:
            print(f"Loading dataset from: {dataset_path}")
        # Try loading from local disk (Arrow format)
        dataset = load_from_disk(dataset_path)
        if split and split in dataset:
            dataset = dataset[split]
        # Limit samples if specified
        if num_samples[idx] is not None:
            dataset = dataset.select(range(min(num_samples[idx], len(dataset))))
        if world_size > 1:
            total_samples = len(dataset)
            samples_per_process = total_samples // world_size
            start_idx = rank * samples_per_process
            end_idx = start_idx + samples_per_process if rank < world_size - 1 else total_samples
            dataset = dataset.select(range(start_idx, end_idx))
            
            if is_main_process:
                print(f"Total samples: {total_samples}, samples per process: {samples_per_process}")
        else:
            total_samples = len(dataset)
            start_idx = 0
        if is_main_process:
            print(f"Processing dataset...")
        
        iterator = tqdm(enumerate(dataset), desc=f"Evaluating (Rank {rank})", disable=not is_main_process)

        dataset_name = None
        if 'journeydb' in dataset_path.lower():
            dataset_name = 'JourneyDB' 
        elif 'laion' in dataset_path.lower():
            dataset_name = 'LAION'
        elif 'blip3o' in dataset_path.lower():
            dataset_name = 'BLIP3o'
        else:
            dataset_name = 'GenAI-Bench-527'
        ori_vocabe = len(tokenizer)
        vocab_size = vqllm.config.vocab_size
        logit_bias = torch.full((vocab_size,), float('-inf'))  # Mask all tokens
        logit_bias[ori_vocabe:vocab_size] = 0
        logit_bias_text = torch.full((vocab_size,), float('-inf'))
        logit_bias_text[:ori_vocabe] = 0
        do_force_image_tokens = False
        do_force_text_tokens = False
        with torch.no_grad():
            for sample_idx, sample in iterator:
                # Extract text
                if sample['data_type']  in ['image_text'] :
                    vqcode = json.loads(sample['vqcode_{}'.format(str(vq_resolution))])
                    vqcode = torch.tensor(vqcode) + ori_vocab_size
                    do_T2I = sample_idx + start_idx < int(T2I_ratio * total_samples)
                    do_uncond_T2I = sample_idx + start_idx < int(T2I_ratio*0.1 * total_samples)
                    if "genai-bench-527" in dataset_path.lower():
                        do_T2I = True
                        do_uncond_T2I = False
                        task_ids = prompt_to_skills[sample['text']]
                        task_ids = [tid.replace(" ", "_") for tid in task_ids]
                    else:
                        task_ids = []
                    if do_T2I:
                        modality_id = name2id[dataset_name+"_T2I" if not do_force_image_tokens else dataset_name+"_T2I_force"]
                        prompt = ' Generate an image based on this description.'
                        text = sample['text']+prompt
                        if do_uncond_T2I:
                            text = "<unconditional>"
                            modality_id = name2id[dataset_name+"_Image" if not do_force_image_tokens else dataset_name+"_Image_force"]
                        text = text+'<boi><eoi>'+tokenizer.eos_token
                        conversations = [text]
                        input_ids = tokenizer(
                            conversations,
                            return_tensors="pt",
                            padding="longest", 
                            truncation=False,
                        ).input_ids[0]
                        instruction_len = len(input_ids[:-2])
                        input_ids = torch.cat([input_ids[:-2],vqcode,input_ids[-2:]])
                        
                        # Get logits and calculate negative log likelihood
                        targets = input_ids.clone()
                        targets[: instruction_len] = IGNORE_INDEX
                        if do_force_image_tokens:
                            targets[-2:] = IGNORE_INDEX  # Ignore eoi and eos token
                            outputs = vqllm(
                                input_ids=input_ids.unsqueeze(0).to(vqllm.device), 
                                attention_mask=torch.ones_like(input_ids).unsqueeze(0).to(vqllm.device),
                                labels=targets.unsqueeze(0).to(vqllm.device),
                                return_dict=True,
                                output_attentions=False,
                                output_hidden_states=False,
                            )
                            logits = outputs.logits  # shape: [batch, seq_len, vocab_size]
                            shift_labels = nn.functional.pad(targets.unsqueeze(0), (0, 1), value=-100)
                            shift_labels = shift_labels[..., 1:].contiguous()
                            # print("Output Loss:", outputs.loss.item() * (input_ids.shape[0] - instruction_len - 2))
                            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='sum')
                            # ori_loss = loss_fct(logits.view(-1, logits.size(-1)).float(), shift_labels.view(-1).to(logits.device)).item()
                            # print("Original Loss:", ori_loss)
                            logits = logits + logit_bias.to(logits.device).unsqueeze(0).unsqueeze(0)
                            loss = loss_fct(logits.view(-1, logits.size(-1)).float(), shift_labels.view(-1).to(logits.device))
                            nll = loss.item()
                            total_nll[modality_id] += nll
                            total_tokens[modality_id] += input_ids.shape[0] - instruction_len - 2
                            num_processed[modality_id] += 1
                            total_bytes[modality_id] += vq_resolution*vq_resolution*3
                        else:
                            outputs = vqllm(
                                input_ids=input_ids.unsqueeze(0).to(vqllm.device), 
                                attention_mask=torch.ones_like(input_ids).unsqueeze(0).to(vqllm.device),
                                labels=targets.unsqueeze(0).to(vqllm.device))
                            nll = outputs.loss.item() * (input_ids.shape[0] - instruction_len)
                            total_nll[modality_id] += nll
                            total_tokens[modality_id] += input_ids.shape[0] - instruction_len
                            num_processed[modality_id] += 1
                            total_bytes[modality_id] += vq_resolution*vq_resolution*3 + len(("<eoi>"+tokenizer.eos_token).encode('utf-8'))
                            for task_id in task_ids:
                                modality_id_task = name2id["GenAI-Bench-527_T2I_"+task_id]
                                total_bytes[modality_id_task] += vq_resolution*vq_resolution*3 + len(("<eoi>"+tokenizer.eos_token).encode('utf-8'))
                                total_nll[modality_id_task] += nll
                                total_tokens[modality_id_task] += input_ids.shape[0] - instruction_len
                                num_processed[modality_id_task] += 1

                    else:
                        modality_id = name2id[dataset_name+"_Caption" if not do_force_text_tokens else dataset_name+"_Caption_force"]
                        caption = sample['text']+tokenizer.eos_token
                        instruction = '<boi><eoi>The caption of this image is:'

                        caption_ids = tokenizer( caption,  return_tensors="pt", padding="longest",  truncation=False, ).input_ids[0]
                        instruct_id = tokenizer( instruction,  return_tensors="pt", padding="longest",  truncation=False, ).input_ids[0]

                        input_ids = torch.cat([instruct_id[:1],
                                           vqcode,
                                           instruct_id[1:],
                                           caption_ids])
                        instruction_len = len(input_ids) - len(caption_ids)
                        total_bytes[modality_id] += len(caption.encode('utf-8'))
                        targets = input_ids.clone()
                        targets[: instruction_len] = IGNORE_INDEX
                        # Get logits and calculate negative log likelihood
                        if do_force_text_tokens:
                            outputs = vqllm(
                                input_ids=input_ids.unsqueeze(0).to(vqllm.device), 
                                attention_mask=torch.ones_like(input_ids).unsqueeze(0).to(vqllm.device),
                                labels=targets.unsqueeze(0).to(vqllm.device),
                                return_dict=True,
                                output_attentions=False,
                                output_hidden_states=False,
                            )
                            logits = outputs.logits  # shape: [batch, seq_len, vocab_size]
                            shift_labels = nn.functional.pad(targets.unsqueeze(0), (0, 1), value=-100)
                            shift_labels = shift_labels[..., 1:].contiguous()
                            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='sum')
                            logits = logits + logit_bias_text.to(logits.device).unsqueeze(0).unsqueeze(0)
                            loss = loss_fct(logits.view(-1, logits.size(-1)).float(), shift_labels.view(-1).to(logits.device))
                            nll = loss.item()
                        else:
                            outputs = vqllm(
                                input_ids=input_ids.unsqueeze(0).to(vqllm.device), 
                                attention_mask=torch.ones_like(input_ids).unsqueeze(0).to(vqllm.device),
                                labels=targets.unsqueeze(0).to(vqllm.device))
                            nll = outputs.loss.item() * (input_ids.shape[0] - instruction_len)
                        total_nll[modality_id] += nll
                        total_tokens[modality_id] += input_ids.shape[0] - instruction_len
                        num_processed[modality_id] += 1
                        for task_id in task_ids:
                            modality_id_task = name2id["GenAI-Bench-527_Caption_"+task_id]
                            total_bytes[modality_id_task] += len(caption.encode('utf-8'))
                            total_nll[modality_id_task] += nll
                            total_tokens[modality_id_task] += input_ids.shape[0] - instruction_len
                            num_processed[modality_id_task] += 1

                        if sample['text'] == '':
                            continue
                        input_ids = caption_ids
                        modality_id = name2id[dataset_name+"_Text" if not do_force_text_tokens else dataset_name+"_Text_force"]
                        # Get logits and calculate negative log likelihood
                        if do_force_text_tokens:
                            targets = input_ids.clone()
                            outputs = vqllm(
                                input_ids=input_ids.unsqueeze(0).to(vqllm.device), 
                                attention_mask=torch.ones_like(input_ids).unsqueeze(0).to(vqllm.device),
                                labels=input_ids.unsqueeze(0).to(vqllm.device),
                                return_dict=True,
                                output_attentions=False,
                                output_hidden_states=False,
                            )
                            logits = outputs.logits  # shape: [batch, seq_len, vocab_size]
                            shift_labels = nn.functional.pad(targets.unsqueeze(0), (0, 1), value=-100)
                            shift_labels = shift_labels[..., 1:].contiguous()
                            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='sum')
                            logits = logits + logit_bias_text.to(logits.device).unsqueeze(0).unsqueeze(0)
                            loss = loss_fct(logits.view(-1, logits.size(-1)).float(), shift_labels.view(-1).to(logits.device))
                            nll = loss.item()
                        else:
                            outputs = vqllm(
                                input_ids=input_ids.unsqueeze(0).to(vqllm.device), 
                                attention_mask=torch.ones_like(input_ids).unsqueeze(0).to(vqllm.device),
                                labels=input_ids.unsqueeze(0).to(vqllm.device))
                            nll = outputs.loss.item() * input_ids.shape[0]
                        # if math.isnan(nll):
                        #     print(caption)
                        #     print(outputs.loss)
                        total_nll[modality_id] += nll
                        total_tokens[modality_id] += input_ids.shape[0]
                        num_processed[modality_id] += 1
                        total_bytes[modality_id] += len(caption.encode('utf-8'))
                        
                else:
                    modality_id = name2id["Text" if not do_force_text_tokens else "Text_force"]
                    text = sample[text_field]+tokenizer.eos_token  
                    assert text != 'no'
                    
                    # Tokenize
                    inputs = tokenizer(
                        text,
                        return_tensors='pt',
                        padding="longest", 
                        truncation=True,             # TODO: should we truncate?
                        max_length=max_length,
                    ).to(vqllm.device)
                    
                    # Get logits and calculate negative log likelihood
                    if do_force_text_tokens:
                        targets = inputs['input_ids'].clone()
                        outputs = vqllm(**inputs, labels=inputs['input_ids'],
                                return_dict=True,
                                output_attentions=False,
                                output_hidden_states=False,)
                        logits = outputs.logits  # shape: [batch, seq_len, vocab_size]
                        shift_labels = nn.functional.pad(targets.unsqueeze(0), (0, 1), value=-100)
                        shift_labels = shift_labels[..., 1:].contiguous()
                        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='sum')
                        logits = logits + logit_bias_text.to(logits.device).unsqueeze(0).unsqueeze(0)
                        loss = loss_fct(logits.view(-1, logits.size(-1)).float(), shift_labels.view(-1).to(logits.device))
                        nll = loss.item()
                    else:
                        outputs = vqllm(**inputs, labels=inputs['input_ids'])
                        nll = outputs.loss.item() * inputs['input_ids'].shape[1]

                    total_nll[modality_id] += nll
                    total_tokens[modality_id] += inputs['input_ids'].shape[1]
                    num_processed[modality_id] += 1

                    # Calculate byte length
                    detokenized_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
                    byte_length = len(detokenized_text.encode('utf-8'))    # TODO: how many bytes should we use for eos tokens?
                    total_bytes[modality_id] += byte_length

    if world_size > 1:
        flat = (
            list(map(float, total_nll))
            + list(map(float, total_bytes))
            + list(map(float, total_tokens))
            + list(map(float, num_processed))
        )
        local_stats = torch.tensor(flat, dtype=torch.float64, device=vqllm.device)
        dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)
        stats = local_stats.cpu().numpy().astype(float).tolist()
    else:
        stats = (
            list(map(float, total_nll))
            + list(map(float, total_bytes))
            + list(map(float, total_tokens))
            + list(map(float, num_processed))
        )

    # Unpack stats into per-modality lists
    nlls = stats[0:datasets_num]
    bytes_list = [int(x) for x in stats[datasets_num:2*datasets_num]]
    tokens_list = [float(x) for x in stats[2*datasets_num:3*datasets_num]]
    proc_list = [int(x) for x in stats[3*datasets_num:4*datasets_num]]

    # Compute per-modality metrics, guarding divide-by-zero
    bits_per_token_list = []
    bits_per_byte_list = []
    perplexity_list = []
    for i in range(datasets_num):
        nll = nlls[i]
        tot_tokens = tokens_list[i]
        tot_bytes = bytes_list[i]

        if tot_tokens > 0:
            bpt = (nll / tot_tokens) / math.log(2)
            ppl = math.exp(nll / tot_tokens)
        else:
            bpt = float("nan")
            ppl = float("inf")

        if tot_bytes > 0:
            bpb = (nll / math.log(2)) / tot_bytes
        else:
            bpb = float("nan")

        bits_per_token_list.append(bpt)
        bits_per_byte_list.append(bpb)
        perplexity_list.append(ppl)

    results = {
            id2name[i]: {
                "bits_per_byte": bits_per_byte_list[i],
                "bits_per_token": bits_per_token_list[i],
                "perplexity": perplexity_list[i],
                "total_bytes": bytes_list[i],
                "total_tokens": tokens_list[i],
                "num_samples": proc_list[i],
            }
            for i in range(len(bits_per_byte_list)) if tokens_list[i] > 0
    }
    
    # Print results only on main process
    if is_main_process:
        if "acheckpoint" in LLM_pth:
            num_of_step = int(LLM_pth.split('-')[-1])
            # num_of_step = num_of_step * 1.25
            json_path = REPO_ROOT + "/data/results/bpb_{}_a{}.json".format(model_name, int(num_of_step))
        else:
            json_path = REPO_ROOT + "/data/results/bpb_{}_{}.json".format(model_name, LLM_pth.split('-')[-1])
        print(f"Saving results to {json_path}")

        # Also write machine-readable JSON (handle torch/numpy scalars)
        serializable = {k: (v.item() if hasattr(v, "item") else v) for k, v in results.items()}
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as jf:
                existing = json.load(jf)
            existing.update(serializable)
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(existing, jf, indent=2)
        else:
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(serializable, jf, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint name')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    args = parser.parse_args()
    ckpt = args.ckpt
    model_name = args.model_name

    tokenizer_name = model_name.split('_qwen3')[0]
    evaluate_bpb(
        model_name=model_name,
        LLM_pth=CKPT_ROOT + '/{}/{}'.format(model_name, ckpt),
        dataset_paths=[
            DATA_ROOT + '/dclm30m_val_hf',
            DATA_ROOT + '/journeydb_val_tokenized_{}_hf'.format(tokenizer_name), 
            DATA_ROOT + '/laion_val_tokenized_{}_hf'.format(tokenizer_name),
            DATA_ROOT + '/blip3o_val_short_tokenized_{}_hf'.format(tokenizer_name),
            # DATA_ROOT + '/GenAI-Bench-527_tokenized_{}'.format(tokenizer_name),
            ],  # Path to Arrow dataset directory
        split='val',
        max_length=2048,
        num_samples=[
            50000,
            6133, 
            39368,
            4499,
            # 3162
            ],  # Optional: limit samples
        vq_resolution=256,
    )
    cleanup_distributed()

# torchrun --nproc_per_node=8 bpb.py --ckpt checkpoint-52740 --model_name gigatok_qwen3_0.6b_mixpretrain_stage1_1_8_0.8_60b_1e-4_1024