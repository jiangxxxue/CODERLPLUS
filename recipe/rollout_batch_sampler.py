import random
from typing import Iterator, List, Optional
import torch
import ray
from torch.utils.data import Sampler
from omegaconf import DictConfig

from .rollout_buffer import get_global_rollout_buffer


class RolloutBatchSampler(Sampler):
    """
    Batch sampler that controls the ratio of original vs rollout data within each batch.
    
    This sampler creates batches where:
    - A fixed ratio of samples come from original dataset (code generation tasks)
    - The remaining samples come from rollout buffer (execution semantics alignment tasks)
    - Epoch is defined by the original dataset being fully traversed
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        min_rollout_samples: int,
        rollout_config: DictConfig,
        drop_last: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Initialize the rollout batch sampler.
        
        Args:
            dataset: The RolloutEnhancedDataset instance
            batch_size: Total batch size
            rollout_config: Configuration for rollout integration
            drop_last: Whether to drop the last incomplete batch
            generator: Random generator for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.generator = generator
        
        self.enable_rollout = rollout_config.get("enable", False)
        self.rollout_ratio = rollout_config.get("ratio", 0.5)
        if self.rollout_ratio > 1.0 or self.rollout_ratio < 0:
            raise Exception("rollout_ratio should range from 0 to 1")
        self.min_rollout_samples = min_rollout_samples
        
        self.rollout_samples_per_batch = int(batch_size * self.rollout_ratio)
        self.original_samples_per_batch = batch_size - self.rollout_samples_per_batch
        
        if hasattr(dataset, '__class__') and hasattr(dataset.__class__.__bases__[0], '__len__'):
            self.original_length = dataset.__class__.__bases__[0].__len__(dataset)
        else:
            self.original_length = len(dataset)
        
        if self.enable_rollout:
            self.rollout_buffer = get_global_rollout_buffer()
        else:
            self.rollout_buffer = None
            
        print(f"RolloutBatchSampler initialized:")
        print(f"  Batch size: {batch_size}")
        print(f"  Original samples per batch: {self.original_samples_per_batch}")
        print(f"  Rollout samples per batch: {self.rollout_samples_per_batch}")
        print(f"  Original dataset length: {self.original_length}")
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches with controlled original/rollout ratio."""
        
        if self.generator is not None:
            g = torch.Generator()
            g.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            g = None
            
        original_indices = torch.randperm(self.original_length, generator=g).tolist()
        
        num_batches = len(original_indices) // self.original_samples_per_batch
        if not self.drop_last and len(original_indices) % self.original_samples_per_batch != 0:
            num_batches += 1
            
        for batch_idx in range(num_batches):
            batch_indices = []
            start_idx = batch_idx * self.original_samples_per_batch
            end_idx = min(start_idx + self.original_samples_per_batch, len(original_indices))
            original_batch_indices = original_indices[start_idx:end_idx]
            batch_indices.extend(original_batch_indices)
            
            actual_rollout_samples = 0
            
            if (self.enable_rollout and
                self.rollout_buffer is not None and
                ray.get(self.rollout_buffer.size.remote()) >= self.min_rollout_samples):

                current_batch_size = len(batch_indices)
                if current_batch_size < self.batch_size:
                    remaining_slots = self.batch_size - current_batch_size
                    rollout_samples_needed = min(self.rollout_samples_per_batch, remaining_slots)
                else:
                    rollout_samples_needed = self.rollout_samples_per_batch

                rollout_buffer_size = ray.get(self.rollout_buffer.size.remote())
                if self.generator is not None:
                    rand_gen = random.Random(self.generator.initial_seed() + batch_idx + 10000)
                    rollout_idxs = rand_gen.choices(range(rollout_buffer_size), k=rollout_samples_needed)
                else:
                    rollout_idxs = random.choices(range(rollout_buffer_size), k=rollout_samples_needed)
                rollout_indices = [self.original_length + i for i in rollout_idxs]
                batch_indices.extend(rollout_indices)
                actual_rollout_samples = len(rollout_indices)
            else:
                needed_samples = self.batch_size - len(batch_indices)
                if needed_samples > 0:
                    if self.generator is not None:
                        rand_gen = random.Random(self.generator.initial_seed() + batch_idx + 20000)
                        additional_indices = rand_gen.choices(original_indices, k=needed_samples)
                    else:
                        additional_indices = random.choices(original_indices, k=needed_samples)
                    batch_indices.extend(additional_indices)
            
            actual_original_samples = len(batch_indices) - actual_rollout_samples
            rollout_ratio = actual_rollout_samples / len(batch_indices) if len(batch_indices) > 0 else 0
            print(f"[DEBUG] Batch {batch_idx}: {actual_original_samples} original + {actual_rollout_samples} rollout "
                  f"(ratio: {rollout_ratio:.2f}, total: {len(batch_indices)})")
            
            if len(batch_indices) > 0:
                if self.generator is not None:
                    random.Random(self.generator.initial_seed() + batch_idx).shuffle(batch_indices)
                else:
                    random.shuffle(batch_indices)
                    
                yield batch_indices
    
    def __len__(self) -> int:
        """Return the number of batches in an epoch."""
        if self.drop_last:
            return self.original_length // self.original_samples_per_batch
        else:
            return (self.original_length + self.original_samples_per_batch - 1) // self.original_samples_per_batch


class RolloutBatchSamplerWrapper:
    """
    Wrapper to make RolloutBatchSampler compatible with PyTorch DataLoader.
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        min_rollout_samples: int,
        rollout_config: DictConfig,
        drop_last: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        self.batch_sampler = RolloutBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            min_rollout_samples=min_rollout_samples,
            rollout_config=rollout_config,
            drop_last=drop_last,
            generator=generator,
        )
    
    def __iter__(self):
        return iter(self.batch_sampler)
    
    def __len__(self):
        return len(self.batch_sampler)
