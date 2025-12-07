from typing import Any, Dict
import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo.metric_utils import compute_data_metrics


def compute_data_metrics_by_source(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    data_sources = batch.non_tensor_batch.get("data_source", None)
    if data_sources is None:
        return compute_data_metrics(batch, use_critic)
    
    exec_semantics_align_mask = np.array([("exec_semantics_align" in ds) for ds in data_sources])
    code_generation_mask = ~exec_semantics_align_mask  # All non-exec_semantics_align tasks
    
    metrics = {}
    
    if code_generation_mask.any():
        cg_metrics = _compute_metrics_with_mask(batch, code_generation_mask, use_critic)
        metrics.update(cg_metrics)
    
    if exec_semantics_align_mask.any():
        cr_metrics = _compute_metrics_with_mask(batch, exec_semantics_align_mask, use_critic)
        cr_prefixed_metrics = {f"train/exec_semantics_align/{k}": v for k, v in cr_metrics.items()}
        metrics.update(cr_prefixed_metrics)
    
    return metrics


def _compute_metrics_with_mask(batch: DataProto, mask: np.ndarray, use_critic: bool = True) -> Dict[str, Any]:
    """
    Computes metrics for a subset of the batch identified by the mask.

    Args:
        batch: A DataProto object containing batch data
        mask: Boolean mask to select subset of samples
        use_critic: Whether to include critic-specific metrics

    Returns:
        A dictionary of metrics for the masked subset
    """
    subset_indices = np.where(mask)[0]
    
    if len(subset_indices) == 0:
        return {}
    
    sequence_score = batch.batch["token_level_scores"].sum(-1)[subset_indices]
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)[subset_indices]
    
    advantages = batch.batch["advantages"][subset_indices]
    returns = batch.batch["returns"][subset_indices]
    
    max_response_length = batch.batch["responses"].shape[-1]
    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()[subset_indices]
    response_mask = batch.batch["response_mask"].bool()[subset_indices]
    
    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()
    
    max_prompt_length = prompt_mask.size(-1)
    
    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)
    
    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    
    if use_critic:
        values = batch.batch["values"][subset_indices]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)
        
        critic_metrics = {
            # values
            "critic/values/mean": torch.mean(valid_values).detach().item(),
            "critic/values/max": torch.max(valid_values).detach().item(),
            "critic/values/min": torch.min(valid_values).detach().item(),
            # vf explained var
            "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        }
        metrics.update(critic_metrics)
    
    if "__num_turns__" in batch.non_tensor_batch:
        num_turns = batch.non_tensor_batch["__num_turns__"][subset_indices]
        metrics.update({
            "num_turns/min": num_turns.min(),
            "num_turns/max": num_turns.max(),
            "num_turns/mean": num_turns.mean(),
        })
    
    return metrics
