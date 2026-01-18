import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
import time
from typing import Any, Dict, List, Tuple
from concurrent.futures import as_completed

import numpy as np
import ray
from tqdm import tqdm

from verl import DataProto

from .rollout_buffer import get_global_rollout_buffer
from .filter import SimpleCodeFilter

# Try to use loky (ProcessPoolExecutor with cloudpickle) for better pickling support
try:
    from loky import get_reusable_executor
    LOKY_AVAILABLE = True
except ImportError:
    from concurrent.futures import ProcessPoolExecutor
    LOKY_AVAILABLE = False
    print("Warning: loky not available for code_extractor. Install with: pip install loky")
    print("Falling back to standard ProcessPoolExecutor")

MAX_TEST_CASES = 1


class CodeExecutor:
    _CAPTURE_TEMPLATE = '''
# Get variables
__v__ = {k: v for k, v in locals().items() 
         if not k.startswith('_') 
         and not callable(v)
         and type(v).__module__ in ('builtins', '__main__')
         and type(v).__name__ in ('int', 'float', 'str', 'bool', 'list', 'dict', 'tuple', 'set', 'NoneType')}

# Convert sets to lists for JSON
for k, v in __v__.items():
    if isinstance(v, set):
        __v__[k] = list(v)

import sys, json
print("___V___", file=sys.stderr)
print(json.dumps(__v__), file=sys.stderr)
'''

    @staticmethod
    def execute_code(code: str, input_data: str, variables_to_track: list = None, timeout: int = 5) -> tuple:
        """
        Fast execution of code with variable tracking.
        
        Args:
            code: Python code to execute
            input_data: Input data for the code
            variables_to_track: List of variable names to track (if None, tracks all simple types)
            timeout: Execution timeout in seconds
            
        Returns:
            tuple: (output/error_message, dict of variable values)
        """
        
        has_main = 'if __name__' in code and '__main__' in code
        
        if variables_to_track:
            track_filter = f" and k in {variables_to_track}"
            capture = CodeExecutor._CAPTURE_TEMPLATE.replace(
                "and type(v).__name__", 
                f"{track_filter} and type(v).__name__"
            )
        else:
            capture = CodeExecutor._CAPTURE_TEMPLATE
        
        if has_main:
            lines = code.splitlines()
            indent = '    '
            
            insert_pos = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() and not lines[i].strip().startswith('#'):
                    insert_pos = i + 1
                    break
            
            capture_lines = [indent + line for line in capture.splitlines()]
            lines[insert_pos:insert_pos] = capture_lines
            wrapper_code = '\n'.join(lines)
        else:
            wrapper_code = f"{code}\n\n{capture}"
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(wrapper_code)
                code_path = f.name
            
            result = subprocess.run(
                [sys.executable, code_path],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            output = result.stdout.strip()
            stderr = result.stderr
            
            variables = {}
            if '___V___' in stderr:
                try:
                    idx = stderr.index('___V___') + 7
                    var_json = stderr[idx:].strip().split('\n')[0]
                    if var_json:
                        variables = json.loads(var_json)
                except (ValueError, json.JSONDecodeError):
                    pass
            
            if result.returncode == 0:
                return output, variables
            else:
                error = stderr.split('___V___')[0].strip() if '___V___' in stderr else stderr.strip()
                return f"ERROR: {error}" if error else "ERROR: Unknown error", variables
                
        except subprocess.TimeoutExpired:
            return "ERROR: Timeout", {}
        except Exception as e:
            return f"ERROR: {str(e)}", {}
        finally:
            try:
                os.unlink(code_path)
            except:
                pass


def extract_python_code(response: str) -> List[str]:
    code_blocks = re.findall(r"```python\n(.*?)```", response, re.DOTALL)
    return [code.strip() for code in code_blocks if code.strip()]


def parse_ground_truth(reward_model_data: Dict[str, Any]) -> Dict[str, List[str]]:
    try:
        if isinstance(reward_model_data, dict) and 'ground_truth' in reward_model_data:
            ground_truth_str = reward_model_data['ground_truth']
            if isinstance(ground_truth_str, str):
                ground_truth = json.loads(ground_truth_str)
                return {
                    'inputs': ground_truth.get('inputs', []),
                    'outputs': ground_truth.get('outputs', [])
                }
    except Exception as e:
        print(f"Warning: Failed to parse ground truth: {e}")
    
    return {'inputs': [], 'outputs': []}


def calculate_cyclomatic_complexity(code: str) -> int:
    """
    Calculate cyclomatic complexity of Python code using SimpleCodeFilter.
    """
    if SimpleCodeFilter is None:
        print("Warning: SimpleCodeFilter not available, returning default complexity 1")
        return 1
    
    try:
        filter_instance = SimpleCodeFilter(max_cyclomatic_complexity=1000)  # Set high limit for calculation only
        complexity = filter_instance.calculate_cyclomatic_complexity(code)
        return complexity
    except Exception as e:
        print(f"Warning: Failed to calculate cyclomatic complexity: {e}")
        return 1


def create_exec_semantics_align_prompt(code: str, test_input: str, local_variable_name: str) -> str:
    """
    Create a execution semantics alignment prompt
    """

    reasoning_content = f"""Given the following Python Code and Input, predict:
1) The code's output value (final_output)
2) The final values of the listed local variables at the moment the code outputs

Python code: ```python{code}```
Input: {test_input}
Target local variables: {local_variable_name}

Instructions:
1. First, write a reasoning section explaining step-by-step how the code executes with the given Input
2. Do not include any JSON in the reasoning section
3. On the LAST line only, output a strict JSON object with the required format

Example of final answer (LAST line only):
{{"final_output": 3\n6, "variables": {{"cnt": 2, "buf": [1, 2]}}}}
"""

    return reasoning_content


def create_exec_semantics_align_task_record(rollout_record: Dict[str, Any], max_local_variable_name=3) -> Dict[str, Any]:
    """
    Create a execution semantics alignment training sample from a rollout record.
    """
    code = rollout_record["code"]
    test_input = rollout_record["test_input"]
    expected_local_variable = rollout_record.get("local_variable", "")
    local_variable_name = list(expected_local_variable.keys())
    if len(local_variable_name) > max_local_variable_name:
        local_variable_name = local_variable_name[-max_local_variable_name:]
    expected_local_variable = {name: value for name, value in expected_local_variable.items() if name in local_variable_name}
    reasoning_content = create_exec_semantics_align_prompt(code, test_input, local_variable_name)
    
    chat_prompt = np.array([
        {"role": "user", "content": reasoning_content}
    ])
    
    cyclomatic_complexity = calculate_cyclomatic_complexity(code)
    
    step = rollout_record.get("step", 0)
    test_case_index = rollout_record.get("test_case_index", 0) 
    original_data_source = rollout_record.get("data_source", "unknown")
    problem_id = f"{original_data_source}_{step}_{test_case_index}"
    
    ground_truth = {
        'input': rollout_record.get("test_input", ""),
        'expected_output': rollout_record.get("actual_output", ""),
        'expected_local_variable': expected_local_variable,
        'code': code,
        'problem_id': problem_id,
        'cyclomatic_complexity': cyclomatic_complexity
    }
    
    task_record = {
        'prompt': chat_prompt,
        'reward_model': {
            'ground_truth': json.dumps(ground_truth)
        },
        'data_source': 'exec_semantics_align', 
        'rollout_metadata': {
            'original_step': rollout_record.get("step", 0),
            'original_data_source': rollout_record.get("data_source", "unknown"),
            'test_result': rollout_record.get("test_result", False),
            'test_case_index': rollout_record.get("test_case_index", 0),
        }
    }
    
    return task_record


def check_exec_semantics_align_task_length(task_record: Dict[str, Any], tokenizer, max_prompt_length: int) -> bool:
    """
    Check if a execution semantics alignment task prompt length is within limits
    """
    messages = task_record['prompt']

    raw_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    token_length = len(tokenizer.encode(raw_prompt, add_special_tokens=False))

    return token_length <= max_prompt_length


def _process_single_response(i, data_source, response, reward, reward_model_data,
                             global_steps, min_complexity, max_complexity):
    """
    Worker function to process a single response in parallel.
    Returns: (task_records, stats_dict)
    """
    stats = {
        'total_candidates': 0,
        'filtered_by_correct_reward': 0,
        'filtered_by_complexity': 0,
        'filtered_by_execution_error': 0,
        'actual_output_is_none': 0,
        'processed': 0
    }

    task_records = []

    if "exec_semantics_align" in data_source or "code_execution" in data_source:
        return task_records, stats

    code_blocks = extract_python_code(response)
    if not code_blocks:
        return task_records, stats

    ground_truth = parse_ground_truth(reward_model_data)
    test_inputs = ground_truth['inputs'][:MAX_TEST_CASES]
    expected_outputs = ground_truth['outputs'][:MAX_TEST_CASES]

    if not test_inputs or not expected_outputs:
        return task_records, stats

    code = code_blocks[-1]
    complexity = calculate_cyclomatic_complexity(code)

    executor = CodeExecutor()

    for test_idx, (test_input, expected_output) in enumerate(zip(test_inputs, expected_outputs)):
        stats['total_candidates'] += 1

        if complexity < min_complexity or complexity > max_complexity:
            stats['filtered_by_complexity'] += 1
            continue

        if reward > 0:
            stats['filtered_by_correct_reward'] += 1
            continue

        actual_output = executor.execute_code(code, test_input)

        if actual_output[0].startswith("ERROR:"):
            stats['filtered_by_execution_error'] += 1
            continue

        if actual_output[0] == '':
            stats['actual_output_is_none'] += 1
            continue

        test_result = False

        raw_record = {
            "step": global_steps,
            "data_source": data_source,
            "code": code,
            "test_input": test_input,
            "expected_output": expected_output,
            "actual_output": actual_output[0],
            "local_variable": actual_output[1],
            "test_result": test_result,
            "test_case_index": test_idx,
        }

        try:
            task_record = create_exec_semantics_align_task_record(raw_record)
            task_record['_input_output_key'] = (test_input, actual_output[0])  # For deduplication in main process
            task_records.append(task_record)
            stats['processed'] += 1
        except Exception as e:
            print(f"Warning: Failed to create execution semantics alignment task from rollout record: {e}")
            continue

    return task_records, stats


def extract_code_generation_rollout(batch: DataProto, tokenizer, global_steps: int, save_to_file: bool = False, 
                                  filter_overlong_prompts: bool = True, max_prompt_length: int = 1024,
                                  min_complexity: int = 0, max_complexity: int = 100):
    """Extract Python code from code generation tasks, create execution semantics alignment tasks and save to rollout buffer
    
    Args:
        batch: DataProto containing batch data with responses, data_source, and rewards
        tokenizer: Tokenizer for decoding responses
        global_steps: Current global training step
        save_to_file: Whether to also save records to file for backup/analysis (deprecated)
        filter_overlong_prompts: Whether to filter out overly long prompts
        max_prompt_length: Maximum allowed prompt length in tokens
        min_complexity: Minimum cyclomatic complexity (default: 5)
        max_complexity: Maximum cyclomatic complexity (default: 25)
    """
    try:
        data_sources = batch.non_tensor_batch.get("data_source", [])
        if len(data_sources) == 0:
            return
        
        token_level_scores = batch.batch.get("token_level_scores", None)
        if token_level_scores is None:
            print("Warning: No token_level_scores found in batch")
            return
            
        final_rewards = token_level_scores.sum(dim=-1).cpu().numpy()  # Shape: [batch_size]
        
        reward_model_data = batch.non_tensor_batch.get("reward_model", [])
        
        responses = batch.batch["responses"]
        response_texts = tokenizer.batch_decode(responses, skip_special_tokens=True)

        exec_semantics_align_task_records = []

        # Track seen (input, output) pairs for deduplication within this batch
        seen_input_output_pairs = set()

        stats = {
            'total_candidates': 0,
            'filtered_by_correct_reward': 0,
            'filtered_by_complexity': 0,
            'filtered_by_duplicate': 0,
            'filtered_by_prompt_length': 0,
            'filtered_by_execution_error': 0,
            'actual_output_is_none': 0,
            'added_to_buffer': 0
        }

        # Process responses in parallel
        parallel_start_time = time.time()

        if LOKY_AVAILABLE:
            executor = get_reusable_executor(max_workers=32)
            use_context_manager = False
        else:
            executor = ProcessPoolExecutor(max_workers=32)
            use_context_manager = True

        try:
            # Submit all processing tasks
            futures = {}
            for i, (data_source, response, reward) in enumerate(zip(data_sources, response_texts, final_rewards)):
                reward_data = reward_model_data[i] if i < len(reward_model_data) else {}
                future = executor.submit(
                    _process_single_response,
                    i, data_source, response, reward, reward_data,
                    global_steps, min_complexity, max_complexity
                )
                futures[future] = i

            # Collect results as they complete with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing code responses"):
                try:
                    task_records, worker_stats = future.result()

                    # Aggregate stats
                    for key in worker_stats:
                        stats[key] = stats.get(key, 0) + worker_stats[key]

                    # Process task records with deduplication and prompt length checking
                    for task_record in task_records:
                        input_output_key = task_record.pop('_input_output_key')

                        if input_output_key in seen_input_output_pairs:
                            stats['filtered_by_duplicate'] += 1
                            continue

                        seen_input_output_pairs.add(input_output_key)

                        if filter_overlong_prompts:
                            if not check_exec_semantics_align_task_length(task_record, tokenizer, max_prompt_length):
                                stats['filtered_by_prompt_length'] += 1
                                continue

                        exec_semantics_align_task_records.append(task_record)
                        stats['added_to_buffer'] += 1

                except Exception as e:
                    print(f"Warning: Failed to process response: {e}")
                    continue
        finally:
            # Clean up executor if using ProcessPoolExecutor
            if use_context_manager:
                executor.shutdown(wait=True)

        parallel_time = time.time() - parallel_start_time
        print(f"[DEBUG] Parallel processing completed in {parallel_time:.2f}s for {len(futures)} responses")

        remaining_after_complexity = stats['total_candidates'] - stats['filtered_by_complexity']
        remaining_after_correct = remaining_after_complexity - stats['filtered_by_correct_reward']
        remaining_after_execution = remaining_after_correct - stats['filtered_by_execution_error']
        remaining_after_actual_output = remaining_after_execution - stats['actual_output_is_none']
        remaining_after_duplicate = remaining_after_actual_output - stats['filtered_by_duplicate']
        remaining_after_prompt_length = remaining_after_duplicate - stats['filtered_by_prompt_length']
        
        print(f"[DEBUG] Rollout extraction statistics at step {global_steps}:")
        print(f"  - Total candidates: {stats['total_candidates']}")
        print(f"  - Filtered by complexity (not in {min_complexity}-{max_complexity}): {stats['filtered_by_complexity']} (remaining: {remaining_after_complexity})")
        print(f"  - Filtered by correct reward (reward > 0): {stats['filtered_by_correct_reward']} (remaining: {remaining_after_correct})")
        print(f"  - Filtered by execution error: {stats['filtered_by_execution_error']} (remaining: {remaining_after_execution})")
        print(f"  - Actual output is None: {stats['actual_output_is_none']} (remaining: {remaining_after_actual_output})")
        print(f"  - Filtered by duplicate (input+output): {stats['filtered_by_duplicate']} (remaining: {remaining_after_duplicate})")
        print(f"  - Filtered by prompt length: {stats['filtered_by_prompt_length']} (remaining: {remaining_after_prompt_length})")
        print(f"  - Added to buffer: {stats['added_to_buffer']}")
        
        if exec_semantics_align_task_records:
            try:
                rollout_buffer = get_global_rollout_buffer()
                
                buffer_size_before = ray.get(rollout_buffer.size.remote())
                buffer_stats_before = ray.get(rollout_buffer.get_stats.remote())
                
                ray.get(rollout_buffer.add_records.remote(exec_semantics_align_task_records))
                
                buffer_size_after = ray.get(rollout_buffer.size.remote())
                buffer_stats_after = ray.get(rollout_buffer.get_stats.remote())
                
                print(f"[DEBUG] Buffer update at step {global_steps}:")
                print(f"  - Added {len(exec_semantics_align_task_records)} new execution semantics alignment task records")
                print(f"  - Buffer size: {buffer_size_before} -> {buffer_size_after}")
                print(f"  - Total added ever: {buffer_stats_after.get('total_added', 0)}")
                print(f"  - Test results - Correct: {buffer_stats_after.get('test_results', {}).get('correct', 0)}, "
                      f"Incorrect: {buffer_stats_after.get('test_results', {}).get('incorrect', 0)}")
                
                correct_in_batch = sum(1 for record in exec_semantics_align_task_records 
                                     if record['rollout_metadata']['test_result'])
                incorrect_in_batch = len(exec_semantics_align_task_records) - correct_in_batch
                print(f"  - This batch - Correct: {correct_in_batch}, Incorrect: {incorrect_in_batch}")
                
            except Exception as e:
                print(f"Warning: Failed to add execution semantics alignment task records to rollout buffer: {e}")
        else:
            print(f"[DEBUG] No execution semantics alignment task records generated at step {global_steps}")
    
    except Exception as e:
        print(f"Warning: Failed to extract code generation rollout: {e}")
        traceback.print_exc()
