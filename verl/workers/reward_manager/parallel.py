# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import multiprocessing as mp
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Any, Dict, List, Tuple

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


def _compute_single_reward_worker(item_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function for computing reward for a single sample in a separate process.
    
    Args:
        item_data: Dictionary containing all necessary data for reward computation
        
    Returns:
        Dictionary with computed reward and metadata
    """
    try:
        # Extract data
        prompt_str = item_data['prompt_str']
        response_str = item_data['response_str'] 
        ground_truth = item_data['ground_truth']
        data_source = item_data['data_source']
        extra_info = item_data['extra_info']
        compute_score_fn = item_data['compute_score_fn']
        item_index = item_data['item_index']
        valid_response_length = item_data['valid_response_length']
        
        # Parse ground truth if it's a string
        if isinstance(ground_truth, str):
            try:
                ground_truth = json.loads(ground_truth)
            except json.JSONDecodeError:
                pass  # Keep as string if not valid JSON
        
        # Debug print
        print(f"Worker {item_index}: Starting compute_score for data_source={data_source}")
        
        # Compute score
        score = compute_score_fn(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        
        print(f"Worker {item_index}: Completed compute_score, got score={score}")
        
        # Handle both dict and scalar returns
        if isinstance(score, dict):
            reward = score["score"]
            extra_score_info = score
        else:
            reward = score
            extra_score_info = {"score": score}
        
        return {
            'success': True,
            'item_index': item_index,
            'reward': reward,
            'valid_response_length': valid_response_length,
            'extra_score_info': extra_score_info,
            'prompt_str': prompt_str,
            'response_str': response_str,
            'ground_truth': ground_truth,
            'data_source': data_source
        }
        
    except Exception as e:
        print(f"Worker {item_data.get('item_index', -1)}: Error - {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'item_index': item_data.get('item_index', -1),
            'error': str(e),
            'reward': 0.0,
            'valid_response_length': item_data.get('valid_response_length', 1),
            'extra_score_info': {"score": 0.0, "error": str(e)},
            'prompt_str': item_data.get('prompt_str', ''),
            'response_str': item_data.get('response_str', ''),
            'ground_truth': item_data.get('ground_truth', ''),
            'data_source': item_data.get('data_source', '')
        }


@register("parallel")
class ParallelRewardManager(AbstractRewardManager):
    """
    Parallel reward manager that uses multiprocessing to compute rewards concurrently.
    
    This significantly speeds up reward computation especially for CPU/IO intensive tasks
    like code execution and mathematical verification.
    """

    def __init__(
        self, 
        tokenizer, 
        num_examine, 
        compute_score=None, 
        reward_fn_key="data_source",
        max_workers=None,
        batch_timeout=60,
        single_timeout=30,
        min_parallel_size=4,
        **kwargs
    ) -> None:
        """
        Initialize the ParallelRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print for debugging.
            compute_score: A function to compute the reward score. If None, uses default_compute_score.
            reward_fn_key: The key used to access the data source. Defaults to "data_source".
            max_workers: Maximum number of worker processes. If None, uses min(cpu_count(), 8).
            batch_timeout: Timeout in seconds for the entire batch processing.
            single_timeout: Timeout in seconds for individual sample processing.
            min_parallel_size: Minimum batch size to use parallel processing.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        
        # Parallel processing parameters
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.batch_timeout = batch_timeout
        self.single_timeout = single_timeout 
        self.min_parallel_size = min_parallel_size
        
        # Statistics
        self._total_batches = 0
        self._parallel_batches = 0
        self._total_time = 0.0
        
        print(f"ParallelRewardManager initialized with max_workers={self.max_workers}")

    def _preprocess_batch(self, data: DataProto) -> List[Dict[str, Any]]:
        """
        Preprocess the batch data for parallel processing.
        
        Args:
            data: DataProto object containing the batch data
            
        Returns:
            List of dictionaries, each containing data for one sample
        """
        batch_items = []
        
        for i in range(len(data)):
            data_item = data[i]
            
            # Decode prompt and response
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # Decode to strings
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            # Extract metadata
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns
            
            # Prepare item data for worker
            item_data = {
                'item_index': i,
                'prompt_str': prompt_str,
                'response_str': response_str,
                'ground_truth': ground_truth,
                'data_source': data_source,
                'extra_info': extra_info,
                'compute_score_fn': self.compute_score,
                'valid_response_length': valid_response_length
            }
            
            batch_items.append(item_data)
            
        return batch_items

    def _process_parallel(self, batch_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process batch items in parallel using ThreadPoolExecutor.
        
        Args:
            batch_items: List of preprocessed item data
            
        Returns:
            List of results in the same order as input
        """
        # Determine optimal number of workers for this batch
        actual_workers = min(self.max_workers, len(batch_items))
        
        results = [None] * len(batch_items)  # Preserve order
        
        try:
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(_compute_single_reward_worker, item): item['item_index']
                    for item in batch_items
                }
                
                # Collect results with timeout
                completed_count = 0
                for future in as_completed(future_to_index, timeout=self.batch_timeout):
                    try:
                        result = future.result(timeout=self.single_timeout)
                        index = future_to_index[future]
                        results[index] = result
                        completed_count += 1
                    except Exception as e:
                        index = future_to_index[future]
                        # Create error result
                        results[index] = {
                            'success': False,
                            'item_index': index,
                            'error': str(e),
                            'reward': 0.0,
                            'valid_response_length': batch_items[index]['valid_response_length'],
                            'extra_score_info': {"score": 0.0, "error": str(e)},
                            'prompt_str': batch_items[index]['prompt_str'],
                            'response_str': batch_items[index]['response_str'],
                            'ground_truth': batch_items[index]['ground_truth'],
                            'data_source': batch_items[index]['data_source']
                        }
                        
                print(f"Parallel processing completed: {completed_count}/{len(batch_items)} tasks")
                
        except TimeoutError:
            print(f"Batch processing timeout after {self.batch_timeout}s")
            # Fill remaining None results with default values
            for i, result in enumerate(results):
                if result is None:
                    results[i] = {
                        'success': False,
                        'item_index': i,
                        'error': 'Batch timeout',
                        'reward': 0.0,
                        'valid_response_length': batch_items[i]['valid_response_length'],
                        'extra_score_info': {"score": 0.0, "error": "Batch timeout"},
                        'prompt_str': batch_items[i]['prompt_str'],
                        'response_str': batch_items[i]['response_str'],
                        'ground_truth': batch_items[i]['ground_truth'],
                        'data_source': batch_items[i]['data_source']
                    }
        
        return results

    def _process_sequential(self, batch_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process batch items sequentially (fallback for small batches).
        
        Args:
            batch_items: List of preprocessed item data
            
        Returns:
            List of results in the same order as input
        """
        results = []
        for item in batch_items:
            result = _compute_single_reward_worker(item)
            results.append(result)
        return results

    def _should_use_parallel(self, batch_items: List[Dict[str, Any]]) -> bool:
        """
        Determine whether to use parallel processing based on task types.
        
        Args:
            batch_items: List of preprocessed item data
            
        Returns:
            bool: True if should use parallel processing, False for sequential
        """
        # Check task types in the batch
        data_sources = [item['data_source'] for item in batch_items]
        
        # If batch contains exec_semantics_align tasks, use sequential processing (fast tasks)
        has_exec_semantics_align = any("exec_semantics_align" in ds for ds in data_sources)
        if has_exec_semantics_align:
            return False
        
        # CRITICAL: Avoid multiprocessing nesting! 
        # LiveCodeBench tasks (codecontests, apps, etc.) use internal multiprocessing
        # which causes deadlocks when nested inside our ThreadPoolExecutor
        multiprocess_conflicting_tasks = [
            "codecontests", "apps", "livecodebench", "taco", "codeforces", 
            "humaneval", "mbpp", "code_generation"
        ]
        
        has_multiprocess_conflicts = any(
            any(conflict_task in ds for conflict_task in multiprocess_conflicting_tasks)
            for ds in data_sources
        )
        
        if has_multiprocess_conflicts:
            print(f"Detected tasks that use internal multiprocessing: {data_sources}")
            print("Using sequential processing to avoid multiprocessing deadlocks")
            return False
        
        # For truly safe parallel tasks, use parallel if batch is large enough
        return len(batch_items) >= self.min_parallel_size

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """
        Compute rewards for the batch, using parallel or sequential processing based on task type.
        """
        start_time = time.time()
        self._total_batches += 1
        
        # Check if rm_scores already exist
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        batch_size = len(data)
        
        # Preprocess batch
        batch_items = self._preprocess_batch(data)
        
        # Choose processing method based on task type and batch size
        if self._should_use_parallel(batch_items):
            task_types = set(item['data_source'] for item in batch_items)
            print(f"Using parallel processing for batch_size={batch_size}, task_types={task_types}")
            results = self._process_parallel(batch_items)
            self._parallel_batches += 1
        else:
            task_types = set(item['data_source'] for item in batch_items)
            print(f"Using sequential processing for batch_size={batch_size}, task_types={task_types}")
            results = self._process_sequential(batch_items)
        
        # Build reward tensor and extra info
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}
        
        for result in results:
            i = result['item_index']
            reward = result['reward']
            valid_response_length = result['valid_response_length']
            
            # Set reward at the last token position
            reward_tensor[i, valid_response_length - 1] = reward
            
            # Collect extra info
            if 'extra_score_info' in result:
                for key, value in result['extra_score_info'].items():
                    reward_extra_info[key].append(value)
            
            # Print debug information
            data_source = result['data_source']
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
                
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", result['prompt_str'])
                print("[response]", result['response_str'])
                print("[ground_truth]", result['ground_truth'])
                print("[reward]", reward)
                if not result['success']:
                    print("[error]", result.get('error', 'Unknown error'))
                if 'extra_score_info' in result:
                    for key, value in result['extra_score_info'].items():
                        print(f"[{key}]", value)
        
        # Update statistics
        elapsed_time = time.time() - start_time
        self._total_time += elapsed_time
        
        print(f"Batch processed in {elapsed_time:.2f}s (avg: {self._total_time/self._total_batches:.2f}s)")
        print(f"Parallel usage: {self._parallel_batches}/{self._total_batches} ({100*self._parallel_batches/self._total_batches:.1f}%)")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
