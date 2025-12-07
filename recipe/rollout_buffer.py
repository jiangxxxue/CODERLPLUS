import json
import threading
from collections import deque
from typing import Dict, List, Any, Optional
import numpy as np
import ray

class RolloutBuffer:
    """
    Thread-safe buffer for managing rollout data from code generation tasks.
    
    This buffer stores rollout records and provides sampling functionality
    for integration with the training dataloader.
    """
    
    def __init__(self, max_size: int = 10000, sampling_strategy: str = "recent"):
        self.max_size = max_size
        self.sampling_strategy = sampling_strategy
        self._buffer = deque(maxlen=max_size)
        self._lock = threading.RLock()
        self._stats = {
            "total_added": 0,
            "data_sources": {},
            "test_results": {"correct": 0, "incorrect": 0}
        }
        self._last_batch_size = 0
    
    def add_records(self, records: List[Dict[str, Any]]) -> None:
        """
        Add rollout records to the buffer.
        
        Args:
            records: List of rollout record dictionaries
        """
        with self._lock:
            # Track the size of this batch
            self._last_batch_size = len(records)
            for record in records:
                self._buffer.append(record)
                self._update_stats(record)
    
    def _update_stats(self, record: Dict[str, Any]) -> None:
        """Update internal statistics."""
        self._stats["total_added"] += 1
        
        data_source = record.get("data_source", "unknown")
        self._stats["data_sources"][data_source] = self._stats["data_sources"].get(data_source, 0) + 1
        
        test_result = record.get("test_result", False)
        if test_result:
            self._stats["test_results"]["correct"] += 1
        else:
            self._stats["test_results"]["incorrect"] += 1
    
    def sample_records(self, num_samples: int) -> List[Dict[str, Any]]:
        """
        Sample records from the buffer.
        
        Args:
            num_samples: Number of records to sample
            
        Returns:
            List of sampled rollout records
        """
        with self._lock:
            if len(self._buffer) == 0:
                return []
            
            num_samples = min(num_samples, len(self._buffer))
            
            if self.sampling_strategy == "recent":
                # Sample from the most recently added batch
                # Only use last batch if it has enough samples, otherwise fallback to whole buffer
                if self._last_batch_size >= num_samples:
                    # Last batch has enough samples, sample from it
                    pool_size = min(self._last_batch_size, len(self._buffer))
                    candidate_pool = list(self._buffer)[-pool_size:]
                    
                    # Randomly sample num_samples from the last batch
                    indices = np.random.choice(len(candidate_pool), size=num_samples, replace=False)
                    return [candidate_pool[i] for i in indices]
                else:
                    # Fallback: last batch doesn't have enough samples, sample from the whole buffer
                    return list(self._buffer)[-num_samples:]
            elif self.sampling_strategy == "random":
                # Random sampling
                indices = np.random.choice(len(self._buffer), size=num_samples, replace=False)
                return [self._buffer[i] for i in indices]
            elif self.sampling_strategy == "balanced":
                # Balanced sampling by test results
                correct_records = [r for r in self._buffer if r.get("test_result", False)]
                incorrect_records = [r for r in self._buffer if not r.get("test_result", False)]
                
                # Try to get equal numbers of correct and incorrect samples
                correct_samples = min(num_samples // 2, len(correct_records))
                incorrect_samples = min(num_samples - correct_samples, len(incorrect_records))
                
                # If we don't have enough of one type, take more from the other
                if correct_samples < num_samples // 2 and len(incorrect_records) > incorrect_samples:
                    incorrect_samples = min(num_samples - correct_samples, len(incorrect_records))
                elif incorrect_samples < num_samples // 2 and len(correct_records) > correct_samples:
                    correct_samples = min(num_samples - incorrect_samples, len(correct_records))
                
                sampled_records = []
                if correct_samples > 0:
                    correct_indices = np.random.choice(len(correct_records), size=correct_samples, replace=False)
                    sampled_records.extend([correct_records[i] for i in correct_indices])
                
                if incorrect_samples > 0:
                    incorrect_indices = np.random.choice(len(incorrect_records), size=incorrect_samples, replace=False)
                    sampled_records.extend([incorrect_records[i] for i in incorrect_indices])
                
                return sampled_records
            else:
                raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
    
    def get_all_records(self) -> List[Dict[str, Any]]:
        """Get all records in the buffer."""
        with self._lock:
            return list(self._buffer)
    
    def clear(self) -> None:
        """Clear all records from the buffer."""
        with self._lock:
            self._buffer.clear()
            self._stats = {
                "total_added": 0,
                "data_sources": {},
                "test_results": {"correct": 0, "incorrect": 0}
            }
    
    def size(self) -> int:
        """Get the current size of the buffer."""
        with self._lock:
            return len(self._buffer)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            return dict(self._stats)
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save buffer contents to a JSONL file.
        
        Args:
            filepath: Path to save the file
        """
        with self._lock:
            with open(filepath, 'w', encoding='utf-8') as f:
                for record in self._buffer:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load records from a JSONL file into the buffer.
        
        Args:
            filepath: Path to the JSONL file
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                records = []
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
                self.add_records(records)
        except FileNotFoundError:
            print(f"Rollout file {filepath} not found, starting with empty buffer")
        except Exception as e:
            print(f"Error loading rollout file {filepath}: {e}")

@ray.remote
class SharedRolloutBuffer:
    def __init__(self, max_size: int = 10000, sampling_strategy: str = "recent"):
        self.buffer = RolloutBuffer(max_size, sampling_strategy)

    def add_records(self, records: List[Dict[str, Any]]):
        self.buffer.add_records(records)

    def size(self):
        return self.buffer.size()

    def sample_records(self, num_samples: int):
        return self.buffer.sample_records(num_samples)

    def get_stats(self):
        return self.buffer.get_stats()

    def load_from_file(self, filepath: str):
        return self.buffer.load_from_file(filepath)


def get_global_rollout_buffer(max_size: int = 10000, sampling_strategy: str = "recent"):
    try:
        # Try to get existing named actor
        return ray.get_actor("shared_rollout_buffer")
    except ValueError:
        # Create new named actor if it doesn't exist
        return SharedRolloutBuffer.options(name="shared_rollout_buffer").remote(max_size, sampling_strategy)


def reset_global_rollout_buffer() -> None:
    """Reset the global rollout buffer."""
    try:
        actor = ray.get_actor("shared_rollout_buffer")
        ray.kill(actor)
    except ValueError:
        pass
