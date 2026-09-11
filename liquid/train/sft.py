# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2025 Junfeng Wu
# ------------------------------------------------------------------------
import os
import copy
import random
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import numpy as np
import torch
import transformers
import wandb

from liquid.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from liquid.train.llava_trainer import LLaVATrainer

from liquid import conversation as conversation_lib

from PIL import Image
from transformers import TrainerCallback
from transformers import AutoConfig, AutoModelForCausalLM 
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from datasets import load_from_disk,concatenate_datasets


local_rank = None

class SaveAtSpecificStepsCallback(TrainerCallback):
    def __init__(self, steps_to_save):
        self.steps_to_save = set(steps_to_save)
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.steps_to_save:
            control.should_save = True
            print(f"Saving checkpoint at predefined step {state.global_step}")
        
        return control


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    image_processor: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    t2i_source: str = field(default=None,
                           metadata={"help": "Source name of the T2I training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    vq_resolution: str = '512'
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    cfg_ratio: Optional[float] = field(default=0.9, metadata={"help": "fraction of T2I samples that keep their caption; the remaining 1-cfg_ratio are replaced by <unconditional> for CFG"})
    percentage: Optional[str] = field(default='1.0', metadata={"help": "how many data to use"})
    shuffleseed: Optional[int] = field(default=42)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    lr_multi: Optional[str] = field(default=None)
    label_smoothing_factor: float = 0.0
    from_scratch: bool = False
    text_weight: Optional[float] = None
    save_steps_set: Optional[List[int]] = None

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'vlm_uni']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector', 'vision_resampler', 'vlm_uni']
        # add vision tower
        keys_to_match.extend(['vision_tower'])
        # add vision tower aux
        keys_to_match.extend(['vision_fpn', 'vision_stages'])
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg



def random_choice_t2iprompt_from_list():
    my_list = [
        ' Generate an image based on this description.',
        ' Create an image that captures the provided description.',
        ' Based on the previous text, produce a corresponding image.',
        ' Please illustrate the above text with a picture.',
        ' Translate the given description into a image.',
        ' Construct a visual representation of the above description.',
        ' Create a image that matches the text.',
        ' Formulate a visual expression that reflects the narrative just provided.',
        ' Give a visual depiction based on the above sentences.',
        ' Create an image using the information mentioned above as guidance.',
    ]
    return random.choice(my_list)

@dataclass
class DataCollatorForSFTDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    data_args: None

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        vq_resolution = self.data_args.vq_resolution
        t2i_sources = self.data_args.t2i_source.split('^^')
        cfg_ratio = self.data_args.cfg_ratio
        processed_instances = []
        modality_name2id = {'T2I':1, 'Caption':2, 'Text':3} # Here, Caption is QA
        for sources in instances:
            # data_type == "instruction"
            if sources['source'][0] == '{':
                source_name = json.loads(sources['source'])['images']
            else:
                source_name = sources['source']
            if source_name in t2i_sources: 
                vqcode = json.loads(sources['vqcode_{}'.format(str(vq_resolution))])
                vqcode = torch.tensor(vqcode) + len(self.tokenizer)
                modality_id = 1 # T2I
                prompt = random_choice_t2iprompt_from_list()
                    
                if 'multi' in sources['data_type']: # for multi resolution generation
                    prompt = 'Image Size: Width is {} Height is {}.'.format(sources['width'],sources['height'] )  + prompt
                
                for sentence in sources['text']:
                    if sentence['from'] == 'gpt':
                        text = sentence['value']+prompt 
                        break

                if np.random.rand() > cfg_ratio :
                    if 'multi' in sources['data_type']:
                        text =  'Image Size: Width is {} Height is {}. <unconditional>'.format(sources['width'],sources['height'] ) 
                    else:
                        text = "<unconditional>"
                
                text = text+'<boi><eoi>'+self.tokenizer.eos_token
                conversations = [text]
                input_ids = self.tokenizer(
                        conversations,
                        return_tensors="pt",
                        padding="longest",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                    ).input_ids[0]
                
                instruction_len = len(input_ids[:-2])
                input_ids = torch.cat([input_ids[:-2],vqcode,input_ids[-2:]])
                targets = input_ids.clone()
                targets[: instruction_len] = IGNORE_INDEX
                
            else:
                text = sources['text']
                if sources['vqcode_{}'.format(str(vq_resolution))] == 'no':
                    modality_id = 3 # Text only
                else: 
                    modality_id = 2 # QA
                    contain_default_image_token = False
                    for i, sentence in enumerate(text):
                        if DEFAULT_IMAGE_TOKEN in sentence['value']:
                            contain_default_image_token = True
                            text[i]['value'] = text[i]['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                            text[i]['value'] = '<boi><eoi>' + '\n' + text[i]['value']
                            text[i]['value'] = text[i]['value'].strip()
                            break
                    if not contain_default_image_token:
                        text[0]['value'] = text[0]['value'].strip()
                        text[0]['value'] = '<boi><eoi>' + '\n' + text[0]['value']
                        text[0]['value'] = text[0]['value'].strip()

                conv = conversation_lib.default_conversation.copy()
                roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
                conversations = []
                if roles[text[0]["from"]] != conv.roles[0]:
                    # Skip the first one if it is not from human
                    text = text[1:]

                conv.messages = []
                for j, sentence in enumerate(text):
                    role = roles[sentence["from"]]
                    assert role == conv.roles[j % 2], f"{sources['source']} {text}"
                    conv.append_message(role, sentence["value"])
                prompt = conv.get_prompt()
                conversations.append(prompt)    # Currently we do not add <eos> here
                input_ids = self.tokenizer(
                        conversations,
                        return_tensors="pt",
                        padding="longest",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                    ).input_ids[0]
                if modality_id == 2:
                    vqcode = json.loads(sources['vqcode_{}'.format(str(vq_resolution))])
                    vqcode = torch.tensor(vqcode) + len(self.tokenizer)
                    boi_id = self.tokenizer.convert_tokens_to_ids('<boi>')
                    start_idx = input_ids.tolist().index(boi_id)
                    input_ids = torch.cat([input_ids[:start_idx+1],vqcode,input_ids[start_idx+1:]])
                
                targets = input_ids.clone()
                targets[:] = IGNORE_INDEX

                # Find all "<|im_start|>assistant" positions
                assistant_start_seq = self.tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
                im_end_token_id = self.tokenizer("<|im_end|>", add_special_tokens=False).input_ids[0]

                # Scan through and unmask assistant responses
                i = 0
                while i < len(input_ids) - len(assistant_start_seq):
                    # Check if we found assistant start
                    if input_ids[i:i+len(assistant_start_seq)].tolist() == assistant_start_seq:
                        # Skip the prefix "<|im_start|>assistant\n"
                        i += len(assistant_start_seq)
                        
                        # Unmask until we hit <|im_end|> (inclusive)
                        while i < len(input_ids):
                            targets[i] = input_ids[i]  # Unmask this token
                            
                            # Check if this token IS <|im_end|>
                            if input_ids[i] == im_end_token_id:
                                i += 1
                                break  # Done with this assistant response
                            
                            i += 1
                    else:
                        i += 1
                
                # modified_input_ids = input_ids.clone()
                # boi_id = self.tokenizer.convert_tokens_to_ids('<boi>')
                # modified_input_ids[input_ids >= len(self.tokenizer)] = boi_id
                # input_text = self.tokenizer.decode(modified_input_ids, skip_special_tokens=False)
                # try:
                #     record = {
                #         "input_text": input_text,
                #         "input_ids": input_ids.detach().cpu().tolist(),
                #         "targets": targets.detach().cpu().tolist(),
                #     }
                #     with open("sft_collator_debug.jsonl", "a", encoding="utf-8") as f:
                #         f.write(json.dumps(record) + "\n")
                # except Exception:
                #     logging.exception("Failed to write collator debug")
                        
            modality_ids = torch.ones_like(targets) * modality_id
            dataset_ids = torch.ones_like(targets) * 0 # no dataset_id for now
            modality_ids[targets == IGNORE_INDEX] = IGNORE_INDEX
            dataset_ids[targets == IGNORE_INDEX] = IGNORE_INDEX

            processed_instances.append(dict(
                input_ids=input_ids,
                labels=targets,
                modality_ids=modality_ids,
                dataset_ids=dataset_ids
            ))

                
        ### batching ... 
        # import pdb;pdb.set_trace()
        input_ids, labels, modality_ids, dataset_ids = tuple([instance[key] for instance in processed_instances]
                                for key in ("input_ids", "labels", "modality_ids", "dataset_ids"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        modality_ids = torch.nn.utils.rnn.pad_sequence(
            modality_ids,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        dataset_ids = torch.nn.utils.rnn.pad_sequence(
            dataset_ids,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        modality_ids = modality_ids[:, :self.tokenizer.model_max_length]
        dataset_ids = dataset_ids[:, :self.tokenizer.model_max_length]
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            modality_ids=modality_ids,
            dataset_ids=dataset_ids
        )
        # import pdb;pdb.set_trace()
       

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
    #                             data_path=data_args.data_path,
    #                             data_args=data_args)
    data_path=data_args.data_path
    t2i_source=data_args.t2i_source
    percentage = data_args.percentage
    shuffleseed = data_args.shuffleseed

    if '^^' in data_path: # use ^^ to concat multi datasets
        data_paths = data_path.split('^^')

        if '^^' in percentage:
            percentages = [float(p) for p in percentage.split('^^')]
        else:
            percentages = [float(percentage)] * len(data_paths)
        assert len(percentages) == len(data_paths)

        hgdata_list = []
        print('loading subsets...')
        for percent,hgdata_path in zip(percentages,data_paths):
            subset = load_from_disk(hgdata_path)
            sub_len = subset.num_rows
            subset = subset.select(range(int(sub_len*percent)))
            hgdata_list.append(subset)
        train_dataset = concatenate_datasets(hgdata_list)
        if shuffleseed!= 0:
            print('shuffling...')
            train_dataset = train_dataset.shuffle(seed=shuffleseed)
        print(hgdata_list)
    else:
        print('loading subsets...')
        train_dataset = load_from_disk(data_path)
        sub_len = train_dataset.num_rows
        percentage = float(percentage)
        train_dataset = train_dataset.select(range(int(sub_len*percentage)))
        if shuffleseed!= 0:
            print('shuffling...')
            train_dataset = train_dataset.shuffle(seed=shuffleseed)
        print(train_dataset)

    print('training samples: ',train_dataset.num_rows)


    data_collator = DataCollatorForSFTDataset(tokenizer=tokenizer,data_args=data_args)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

class CacheFlushCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.gradient_accumulation_steps == 0:
            # Synchronize before flushing to ensure all GPUs are ready
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

def train():
    attn_implementation="flash_attention_2"
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))


    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            # load_in_4bit=training_args.bits == 4,
            # load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        **bnb_model_from_pretrained_args
    )
   
    model.config.use_cache = False


    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if "gemma" in model_args.model_name_or_path or "gemma" in model_args.version:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )
    else:
        # fix bugs after special token with use_fast=True
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    elif "gemma" in model_args.version:        
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["gemma"]
    elif "qwen" in model_args.version:
        tokenizer.eos_token = "<|im_end|>"      # NOTE: We change the eos token here
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["chatml_direct"]
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    
    # import pdb;pdb.set_trace()
    
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    if training_args.save_steps_set is not None:
        trainer = LLaVATrainer(model=model,
                        tokenizer=tokenizer,
                        args=training_args,
                        callbacks=[SaveAtSpecificStepsCallback(steps_to_save=training_args.save_steps_set)],
                        **data_module)
    else:
        trainer = LLaVATrainer(model=model,
                        tokenizer=tokenizer,
                        args=training_args,
                        **data_module)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    if trainer.is_world_process_zero():
        if "wandb" in training_args.report_to:
            total_flos = trainer.state.total_flos
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                total_flos *= world_size   # scale to all GPUs
            wandb.log({"train/total_flos": total_flos})

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
