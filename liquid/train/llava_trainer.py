import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
    _is_peft_model,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)
from typing import List, Optional
import wandb
import json


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        # import pdb;pdb.set_trace()
        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.args.lr_multi is not None:
            lr_multi_dict = {}
            for _dict in self.args.lr_multi.split(','):
                _key_val = _dict.split(':')
                print("_key_val:", _key_val)
                lr_multi_dict[_key_val[0]] = float(_key_val[1])

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            elif self.args.lr_multi is not None:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and not any([_key in n for _key in lr_multi_dict.keys()]))
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and not any([_key in n for _key in lr_multi_dict.keys()]))
                        ],
                        "weight_decay": 0.0,
                    },
                ]
                for _key in lr_multi_dict:
                    _key_decay = [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and _key in n)
                        ]
                    _key_no_decay = [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and _key in n)
                        ]
                    print("Params LR Change:", _key, "NUM:", len(_key_decay), len(_key_no_decay))
                    if len(_key_decay) > 0:
                        optimizer_grouped_parameters.append(
                            {
                                "params": _key_decay,
                                "lr": self.args.learning_rate * lr_multi_dict[_key],
                                "weight_decay": self.args.weight_decay,
                            },
                        )
                    if len(_key_no_decay) > 0:
                        optimizer_grouped_parameters.append(
                            {
                                "params": _key_no_decay,
                                "lr": self.args.learning_rate * lr_multi_dict[_key],
                                "weight_decay": 0.0,
                            },
                        )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            keys_to_match.extend(['vlm_att', 'vlm_uni'])
            keys_to_match.extend(['vision_fpn', 'vision_stages', 'vision_tower'])
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        modality_ids, dataset_ids = inputs.pop("modality_ids", None), inputs.pop("dataset_ids", None)
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
            logits = outputs["logits"]
            labels = inputs["labels"]
            shift_labels = nn.functional.pad(labels, (0, 1), value=-100)
            shift_labels = shift_labels[..., 1:].contiguous()

            # Flatten the tokens
            logits = logits.view(-1, model.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(logits.device)
            
            per_token_loss = nn.functional.cross_entropy(logits.float(), shift_labels, ignore_index=-100, reduction="none")
            # # Test
            # train_loss = nn.functional.cross_entropy(logits.float(), shift_labels, ignore_index=-100, reduction="mean")
            # print(per_token_loss)
            # print("Train loss check:", train_loss.item(), loss.item())
            # # There is a small numerical difference due to different reduction methods and limited precision (bf16).
            modality_ids = nn.functional.pad(modality_ids, (0, 1), value=-100)
            dataset_ids = nn.functional.pad(dataset_ids, (0, 1), value=-100)
            modality_ids = modality_ids[..., 1:].contiguous().view(-1).to(logits.device)
            dataset_ids = dataset_ids[..., 1:].contiguous().view(-1).to(logits.device)

            modality_id2name = {1: 'T2I', 2: 'Caption', 3: 'Text', 4: 'T_T2I', 5: 'I_Caption'
                                # 4: 'Debug_T2I'
                                }
            dataset_id2name = {1: 'dclm', 2: 'laion-aesthetics', 3: 'midjourney', 4: 'blip3o-60k-short'}

            modality_losses = {}
            dataset_losses = {}
            modality_counts = {}
            dataset_counts = {}
            for i in range(1, 6):
                modality_mask = (modality_ids == i)
                if modality_mask.sum() > 0:
                    modality_losses[i] = per_token_loss[modality_mask].sum()
                    modality_counts[i] = modality_mask.sum()
                else:
                    modality_losses[i] = torch.zeros((), dtype=per_token_loss.dtype, device=logits.device)
                    modality_counts[i] = torch.zeros((), dtype=torch.long, device=logits.device)

            weighted_loss = torch.zeros((), dtype=per_token_loss.dtype, device=logits.device)
            for i in range(1, 5):
                dataset_mask = (dataset_ids == i)
                if dataset_mask.sum() > 0:
                    dataset_losses[i] = per_token_loss[dataset_mask].sum()
                    dataset_counts[i] = dataset_mask.sum()
                    if self.args.text_weight is not None:
                        weighted_loss += (dataset_losses[i] * self.args.text_weight) if i == 1 else dataset_losses[i] 
                else:
                    dataset_losses[i] = torch.zeros((), dtype=per_token_loss.dtype, device=logits.device)
                    dataset_counts[i] = torch.zeros((), dtype=torch.long, device=logits.device)
            weighted_loss = weighted_loss / sum(dataset_counts.values())

            if self.state.global_step % self.args.logging_steps == 0:
                for i in range(1, 6):
                    all_modality_losses = self.accelerator.gather(modality_losses[i])
                    all_modality_counts = self.accelerator.gather(modality_counts[i])
                    
                    total_modality_loss = all_modality_losses.sum()
                    total_modality_count = all_modality_counts.sum()

                    if total_modality_count > 0 and self.is_world_process_zero():
                        avg_modality_loss = (total_modality_loss / total_modality_count).item()
                        if "wandb" in self.args.report_to:
                            wandb.log({f"train/modality_{modality_id2name[i]}_loss": avg_modality_loss}, step=self.state.global_step)

                for i in range(1, 5):
                    all_dataset_losses = self.accelerator.gather(dataset_losses[i])
                    all_dataset_counts = self.accelerator.gather(dataset_counts[i])
                    
                    total_dataset_loss = all_dataset_losses.sum()
                    total_dataset_count = all_dataset_counts.sum()

                    if total_dataset_count > 0 and self.is_world_process_zero():
                        avg_dataset_loss = (total_dataset_loss / total_dataset_count).item()
                        if "wandb" in self.args.report_to:
                            wandb.log({f"train/dataset_{dataset_id2name[i]}_loss": avg_dataset_loss}, step=self.state.global_step)
        
        if self.args.text_weight is not None:
            loss = weighted_loss

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        # if self.is_world_process_zero():
        #     wandb.log(log_data, step=self.state.global_step)

        return (loss, outputs) if return_outputs else loss
