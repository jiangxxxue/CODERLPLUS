#!/usr/bin/env python3
"""
LiveCodeBench Code Execution Task Evaluation Script
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add the LiveCodeBench path to sys.path
current_dir = Path(__file__).parent
lcb_path = current_dir / "LiveCodeBench-main"
sys.path.insert(0, str(lcb_path))

from lcb_runner.benchmarks.code_execution import load_code_execution_dataset
from lcb_runner.lm_styles import LMStyle
from lcb_runner.utils.extraction_utils import extract_execution_code
from lcb_runner.evaluation.compute_code_execution_metrics import code_execution_metrics

from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on LiveCodeBench Code Execution task")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--release_version", type=str, default="release_v4", 
                       choices=["release_v1", "release_v2", "release_v3", "release_v4"],
                       help="LiveCodeBench release version")
    parser.add_argument("--start_date", type=str, default=None, help="Start date for filtering (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default=None, help="End date for filtering (YYYY-MM-DD)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum tokens to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples per problem")
    return parser.parse_args()


def load_dataset(release_version="release_v4"):
    """Load the Code Execution dataset"""
    try:
        dataset = load_code_execution_dataset(release_version)
        print(f"Loaded {len(dataset)} Code Execution problems")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []


def filter_dataset_by_date(dataset, start_date=None, end_date=None):
    """Filter dataset by contest date - following LiveCodeBench official implementation"""
    if not start_date and not end_date:
        return dataset
        
    # Parse contest_date to datetime objects (following official implementation)
    for problem in dataset:
        if hasattr(problem, 'contest_date') and problem.contest_date:
            if not isinstance(problem.contest_date, datetime):
                try:
                    problem.contest_date = datetime.fromisoformat(problem.contest_date)
                except Exception as e:
                    print(f"Warning: Could not parse date for problem {problem.question_id}: {problem.contest_date}, error: {e}")
                    continue
    
    # Filter by start_date (following official implementation)
    if start_date is not None:
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [
            problem for problem in dataset 
            if hasattr(problem, 'contest_date') and isinstance(problem.contest_date, datetime) 
            and start_datetime <= problem.contest_date
        ]
    
    # Filter by end_date (following official implementation)
    if end_date is not None:
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [
            problem for problem in dataset 
            if hasattr(problem, 'contest_date') and isinstance(problem.contest_date, datetime)
            and problem.contest_date <= end_datetime
        ]
    
    print(f"Filtered dataset: {len(dataset)} problems")
    if start_date:
        print(f"  Start date: {start_date}")
    if end_date:
        print(f"  End date: {end_date}")
        
    return dataset


def extract_answer(model_output):
    import re
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    boxed_match = re.search(boxed_pattern, model_output)
    if boxed_match:
        return boxed_match.group(1)
    
    # Try to extract from common answer patterns at the end of the text
    # Look for patterns like "The answer is X" or "The output is X"
    answer_patterns = [
        r'(?:the\s+(?:final\s+)?(?:answer|output|result)\s+is\s*:?\s*)([^\n]+)',
        r'(?:therefore,?\s+the\s+(?:answer|output|result)\s+is\s*:?\s*)([^\n]+)',
        r'(?:so,?\s+the\s+(?:answer|output|result)\s+is\s*:?\s*)([^\n]+)',
        r'(?:answer\s*:?\s*)([^\n]+)',
        r'(?:output\s*:?\s*)([^\n]+)',
        r'(?:result\s*:?\s*)([^\n]+)',
    ]
    
    # Search from the end of the text backwards for better accuracy
    lines = model_output.strip().split('\n')
    for line in reversed(lines):
        for pattern in answer_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Clean up common prefixes/suffixes and extra punctuation
                answer = re.sub(r'^["\'\`]|["\'\`\.]$', '', answer)
                return answer
    
    # Look for list patterns in the last few lines
    list_pattern = r'(\[.*?\])'
    for line in reversed(lines[-5:]):  # Check last 5 lines
        match = re.search(list_pattern, line)
        if match:
            return match.group(1)
    
    # Look for number patterns in the last few lines
    number_pattern = r'\b(-?\d+(?:\.\d+)?)\b'
    for line in reversed(lines[-3:]):  # Check last 3 lines
        matches = re.findall(number_pattern, line)
        if matches:
            return matches[-1]  # Return the last number found
    
    # If no clear pattern found, return the last non-empty line
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith(('Let', 'I', 'We', 'The function', 'Step')):
            return line
    
    # Final fallback - return the last line
    return lines[-1].strip() if lines else ""


def make_conv_hf(problem, tokenizer):
    try:
        system_prompt_path = "benchmark_evaluation/eval/system_prompt.md"
        with open(system_prompt_path, 'r') as f:
            system_prompt = f.read()
    except:
        system_prompt = "You are an expert at Python programming, code execution, test case generation, and fuzzing."
    
    code = problem.code
    test_input = problem.input
    
    reasoning_content = f"""Analyze the following Python code and predict its output for the given input.

```python
{code}
```

Input: {test_input}

Please think step by step and provide your final answer in the format \\boxed{{output}} where output is exactly what the code produces."""
    
    if system_prompt:
        msg = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": reasoning_content}
        ]
    else:
        msg = [
            {"role": "user", "content": reasoning_content}
        ]
    
    chat = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return chat


def generate_sample_batch(question_list, args):
    num_gpus = torch.cuda.device_count()
    
    # For Qwen2.5-Coder-7B-Instruct, use tensor_parallel_size=1 to avoid head divisibility issues
    # The model has 28 attention heads which is not divisible by larger tensor parallel sizes
    tensor_parallel_size = 1 if num_gpus > 1 else 1
    
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
    )
    
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        n=args.n_samples,
        stop=["<|eot_id|>", "</s>"],
    )
    
    outputs = llm.generate(question_list, sampling_params, use_tqdm=True)
    
    all_completions = []
    all_raw_outputs = []
    for output in outputs:
        problem_completions = []
        problem_raw_outputs = []
        for sample_output in output.outputs:
            raw_text = sample_output.text
            extracted_answer = extract_answer(raw_text)
            problem_completions.append(extracted_answer)
            problem_raw_outputs.append(raw_text)
        all_completions.append(problem_completions)
        all_raw_outputs.append(problem_raw_outputs)
    
    return all_completions, all_raw_outputs


def evaluate_model(args):
    os.makedirs(args.save_dir, exist_ok=True)
    
    dataset = load_dataset(args.release_version)
    if not dataset:
        print("Failed to load dataset")
        return

    if args.start_date or args.end_date:
        dataset = filter_dataset_by_date(dataset, args.start_date, args.end_date)
        if not dataset:
            print("No problems remaining after date filtering")
            return

    print(f"Loading tokenizer from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    print(f"Starting evaluation on {len(dataset)} problems...")
    print(f"Using custom reasoning prompt format with \\boxed{{}} output")

    prompts = []
    for problem in tqdm(dataset, desc="Preparing prompts"):
        prompt = make_conv_hf(problem, tokenizer)
        prompts.append(prompt)

    print("Generating responses...")
    all_generations, all_raw_outputs = generate_sample_batch(prompts, args)

    all_results = []
    
    for i, (problem, generations) in enumerate(zip(dataset, all_generations)):
        print(f"Processing problem {i+1}/{len(dataset)}: {problem.question_id}")
        print(f"  Generated outputs: {[gen[:50] + '...' if len(gen) > 50 else gen for gen in generations]}")

        result = {
            "question_id": problem.question_id,
            "problem_id": problem.problem_id,
            "function_name": problem.function_name,
            "code": problem.code,
            "input": problem.input,
            "expected_output": problem.output,
            "generated_outputs": generations,
            "raw_outputs": all_raw_outputs[i],
            "prompt": prompts[i]
        }
        all_results.append(result)

    results_file = os.path.join(args.save_dir, "code_execution_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Raw results saved to {results_file}")

    try:
        samples = [
            {
                "code": problem.code,
                "input": problem.input,
                "output": problem.output
            }
            for problem in dataset
        ]

        metrics, detailed_results = code_execution_metrics(samples, all_generations)

        eval_file = os.path.join(args.save_dir, "code_execution_evaluation.json")
        eval_results = {
            "metrics": metrics,
            "detailed_results": detailed_results,
            "config": {
                "model": args.model,
                "release_version": args.release_version,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "temperature": args.temperature,
                "n_samples": args.n_samples,
                "prompt_format": "custom_reasoning_with_boxed_output"
            }
        }
        
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"Evaluation results saved to {eval_file}")
        print(f"Code Execution Pass@1: {metrics['pass@1']:.2f}%")
  
        summary_file = os.path.join(args.save_dir, "result.txt")
        with open(summary_file, 'w') as f:
            f.write(f"LiveCodeBench Code Execution Evaluation Results\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Release Version: {args.release_version}\n")
            if args.start_date:
                f.write(f"Start Date: {args.start_date}\n")
            if args.end_date:
                f.write(f"End Date: {args.end_date}\n")
            f.write(f"Temperature: {args.temperature}\n")
            f.write(f"Number of Problems: {len(dataset)}\n")
            f.write(f"Pass@1: {metrics['pass@1']:.2f}%\n")
            f.write(f"Prompt Format: Custom reasoning with \\boxed{{}} output\n")
        
        print(f"Summary saved to {summary_file}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Raw results have been saved, but evaluation metrics could not be computed.")


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args)
