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

from collections import defaultdict
from typing import Any
from concurrent.futures import as_completed
import time

import torch
import json
from tqdm import tqdm

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

# Try to use loky (ProcessPoolExecutor with cloudpickle) for better pickling support
try:
    from loky import get_reusable_executor
    LOKY_AVAILABLE = True
except ImportError:
    from concurrent.futures import ProcessPoolExecutor
    LOKY_AVAILABLE = False
    print("Warning: loky not available. Install with: pip install loky")
    print("Falling back to standard ProcessPoolExecutor (may not work with dynamically loaded functions)")


def _process_single_item_worker(i, prompt_str, response_str, ground_truth, data_source, extra_info, compute_score):
    """Standalone worker function for processing a single data item (picklable for multiprocessing)."""
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    # Time the compute_score call
    start_time = time.time()
    score = compute_score(
        data_source=data_source,
        solution_str=response_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    compute_time = time.time() - start_time

    return {
        "index": i,
        "score": score,
        "data_source": data_source,
        "prompt_str": prompt_str,
        "response_str": response_str,
        "ground_truth": ground_truth,
        "compute_time": compute_time,
    }


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", max_workers=32) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
            max_workers: Maximum number of parallel workers. If None, uses default (number of CPUs).
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.max_workers = max_workers  # Number of parallel workers

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}
        compute_times = []

        # Track overall timing
        total_start_time = time.time()

        # Pre-process all items: decode prompts and responses
        preprocessed_items = []
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode in main process
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            preprocessed_items.append({
                "index": i,
                "prompt_str": prompt_str,
                "response_str": response_str,
                "ground_truth": ground_truth,
                "data_source": data_source,
                "extra_info": extra_info,
                "valid_response_length": valid_response_length,
            })

        # Process items in parallel using processes (true parallelism for CPU-bound work)
        # Use loky if available (handles cloudpickle for dynamically loaded functions)
        if LOKY_AVAILABLE:
            executor = get_reusable_executor(max_workers=self.max_workers)
            use_context_manager = False
        else:
            executor = ProcessPoolExecutor(max_workers=self.max_workers)
            use_context_manager = True

        try:
            # Submit all tasks with preprocessed data
            futures = {
                executor.submit(
                    _process_single_item_worker,
                    item["index"],
                    item["prompt_str"],
                    item["response_str"],
                    item["ground_truth"],
                    item["data_source"],
                    item["extra_info"],
                    self.compute_score
                ): item
                for item in preprocessed_items
            }

            # Collect results as they complete with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Computing rewards"):
                result = future.result()
                item = futures[future]

                i = result["index"]
                valid_response_length = item["valid_response_length"]
                score = result["score"]
                data_source = result["data_source"]
                prompt_str = result["prompt_str"]
                response_str = result["response_str"]
                ground_truth = result["ground_truth"]
                compute_time = result["compute_time"]

                # Track compute times
                compute_times.append(compute_time)

                # Process score and update reward_tensor
                if isinstance(score, dict):
                    reward = score["score"]
                    # Store the information including original reward
                    for key, value in score.items():
                        reward_extra_info[key].append(value)
                else:
                    reward = score

                reward_tensor[i, valid_response_length - 1] = reward

                # Print debug info (with thread-safe counter)
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)
                    print("[ground_truth]", ground_truth)
                    if isinstance(score, dict):
                        for key, value in score.items():
                            print(f"[{key}]", value)
                    else:
                        print("[score]", score)
                    print(f"[compute_time] {compute_time:.4f}s")
        finally:
            # Clean up executor if using ProcessPoolExecutor (loky executor is reusable)
            if use_context_manager:
                executor.shutdown(wait=True)

        # Log timing statistics
        total_time = time.time() - total_start_time
        if compute_times:
            avg_compute_time = sum(compute_times) / len(compute_times)
            max_compute_time = max(compute_times)
            min_compute_time = min(compute_times)
            total_compute_time = sum(compute_times)
            print(f"\n[Timing Statistics]")
            print(f"  Total items: {len(compute_times)}")
            print(f"  Total wall time: {total_time:.4f}s")
            print(f"  Total compute time: {total_compute_time:.4f}s")
            print(f"  Average compute time: {avg_compute_time:.4f}s")
            print(f"  Min compute time: {min_compute_time:.4f}s")
            print(f"  Max compute time: {max_compute_time:.4f}s")
            print(f"  Speedup: {total_compute_time/total_time:.2f}x\n")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
