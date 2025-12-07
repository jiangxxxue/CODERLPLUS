import json
import random
from typing import Dict, List, Any, Optional, Union
import torch
import numpy as np
import ray
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset
from .rollout_buffer import get_global_rollout_buffer, RolloutBuffer


class RolloutEnhancedDataset(RLHFDataset):
    """
    Dataset that combines original RLHF data with rollout data from code generation tasks.
    
    This dataset extends RLHFDataset to include synthetic training examples generated
    from rollout data, providing a way to learn from execution feedback.
    """
    
    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        super().__init__(data_files, tokenizer, config, processor)
        
        self.rollout_config = config.get("rollout_integration", {})
        self.enable_rollout = self.rollout_config.get("enable", False)
        self.rollout_ratio = self.rollout_config.get("ratio", 0.5)
        self.rollout_buffer_size = self.rollout_config.get("buffer_size", 10000)
        self.rollout_sampling_strategy = self.rollout_config.get("sampling_strategy", "recent")
        self.rollout_prompt_template = self.rollout_config.get("prompt_template", "default")
        
        batch_size = config['train_batch_size']
        self.min_reasoning_samples = int(batch_size * self.rollout_ratio)
        
        if self.enable_rollout:
            self.rollout_buffer = get_global_rollout_buffer(
                max_size=self.rollout_buffer_size,
                sampling_strategy=self.rollout_sampling_strategy
            )
            
            # Load existing rollout data if available
            rollout_file = self.rollout_config.get("existing_rollout_file", None)
            if rollout_file:
                ray.get(self.rollout_buffer.load_from_file.remote(rollout_file))

            buffer_size = ray.get(self.rollout_buffer.size.remote())
            print(f"RolloutEnhancedDataset initialized with {buffer_size} rollout samples")
        else:
            self.rollout_buffer = None
    
    
    def _get_rollout_samples(self, num_samples: int) -> List[Dict[str, Any]]:
        if not self.enable_rollout or self.rollout_buffer is None:
            return []

        buffer_size = ray.get(self.rollout_buffer.size.remote())
        if buffer_size < self.min_reasoning_samples:
            return []

        rollout_records = ray.get(self.rollout_buffer.sample_records.remote(num_samples))

        return rollout_records
    
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, item):
        original_length = super().__len__()
        
        if not self.enable_rollout or self.rollout_buffer is None:
            return super().__getitem__(item)
            
        buffer_size = ray.get(self.rollout_buffer.size.remote())
        
        # Check if this is a rollout sample index (managed by RolloutBatchSampler)
        if (buffer_size >= self.min_reasoning_samples and item >= original_length):
            
            # DEBUG: Track rollout sample access
            if hasattr(self, '_rollout_access_count'):
                self._rollout_access_count += 1
            else:
                self._rollout_access_count = 1
                
            # Get a single rollout sample
            rollout_samples = self._get_rollout_samples(1)
            if rollout_samples:
                rollout_sample = rollout_samples[0]
                if self._rollout_access_count % 50 == 0:  # Print every 50 accesses
                    print(f"[DEBUG] Accessed rollout sample #{self._rollout_access_count}, "
                          f"buffer size: {buffer_size}, sample from step: {rollout_sample.get('rollout_metadata', {}).get('original_step', 'unknown')}")
                return self._process_rollout_sample(rollout_sample)
            else:
                # Fallback to original data if no rollout samples available
                item = item % original_length
                return super().__getitem__(item)
        
        # Return original dataset sample
        return super().__getitem__(item)
    
    def _process_rollout_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        messages = sample[self.prompt_key]
        
        raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        
        import verl.utils.torch_functional as verl_F
        from verl.utils.model import compute_position_id_with_mask
        
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        
        position_ids = compute_position_id_with_mask(attention_mask)
        
        result = {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "data_source": sample.get("data_source", "rollout_unknown"),
        }
        
        extra_info = sample.get("extra_info", {})
        result["index"] = extra_info.get("index", 0)
        result["tools_kwargs"] = extra_info.get("tools_kwargs", {})
        result["interaction_kwargs"] = extra_info.get("interaction_kwargs", {})
        
        if "reward_model" in sample:
            result["reward_model"] = sample["reward_model"]
        
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Rollout prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")
        
        result["raw_prompt_ids"] = raw_prompt_ids
        result['extra_info'] = extra_info
        result['ability'] = 'rollout sample'
        
        if self.return_raw_chat:
            result["raw_prompt"] = messages
        
        if self.return_full_prompt:
            result["full_prompts"] = raw_prompt
        
        return result       
    
    def get_rollout_stats(self) -> Dict[str, Any]:
        """Get statistics about rollout data integration."""
        if not self.enable_rollout or self.rollout_buffer is None:
            return {"enabled": False}

        buffer_stats = ray.get(self.rollout_buffer.get_stats.remote())
        buffer_size = ray.get(self.rollout_buffer.size.remote())

        return {
            "enabled": True,
            "buffer_size": buffer_size,
            "buffer_stats": buffer_stats,
            "rollout_ratio": self.rollout_ratio,
            "original_dataset_size": super().__len__(),
            "total_dataset_size": self.__len__(),
            "rollout_samples_count": max(0, self.__len__() - super().__len__())
        }
