import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import re
import time
import openai
import argparse
import traceback
import pandas as pd
from tqdm import tqdm
from typing import List
from datasets import Dataset
from utils.data import write_jsonl, read_problems
import json
from transformers import AutoTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="")
parser.add_argument("--save_dir", type=str)
parser.add_argument("--input_data", type=str)
parser.add_argument("--num-samples-per-task", type=int, default=1)
parser.add_argument("--temperature", type=float, default=0.)
args = parser.parse_args()

problems = pd.read_json(args.input_data, lines=True)
STOP_WORDS =["\nassert", "assert"]

from vllm import LLM, SamplingParams
import torch

import re
def match_code(s):
    pattern = r'```python(.*?)```'
    sol = re.findall(pattern, s, re.DOTALL)
    if len(sol) > 0:
        for code_block in reversed(sol):
            if 'def ' in code_block:
                return code_block
        return sol[-1]
    
    pattern = r'```(.*?)```'
    sol = re.findall(pattern, s, re.DOTALL)
    if len(sol) > 0:
        for code_block in reversed(sol):
            if 'def ' in code_block:
                return code_block
        return sol[-1]
    
    return s.split('```')[0]


def generate_sample_batch(question_list):
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.80,
    )
    sampling_params = SamplingParams(max_tokens=8192,
                                    temperature=args.temperature,
                                    n=1,
                                    stop=["<|eot_id|>"],)
    
    outputs = llm.generate(question_list, sampling_params, use_tqdm=True)
    completions = ["```python\n" +  match_code(output.outputs[0].text) + "\n```" for output in outputs]
    outputs = [output.outputs[0].text for output in outputs]
    return completions,outputs


def make_conv_hf(example, tokenizer):
    system_prompt = open("benchmark_evaluation/eval/system_prompt.md").read()
    if system_prompt:
        msg = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["prompt_sft"] + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end."}
        ]
    else:
        msg = [
            {"role": "user", "content": example["prompt_sft"] + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end."}
        ]        
    chat = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return chat 

samples = []
del problems["start_time"]

tokenizer = AutoTokenizer.from_pretrained(args.model)
problems["instruction"] = problems.apply(lambda row: make_conv_hf(row, tokenizer), axis=1)


completions,outputs = generate_sample_batch(problems["instruction"])
for i in range(len(completions)):
    if "class Solution:" not in completions[i]:
        completions[i] = completions[i].replace("```python", "").replace("```", "")
        completions[i] = "\n".join(["    " + line for line in completions[i].split("\n")])
        completions[i] = "class Solution:\n" + completions[i]
        completions[i] = "```python\n" + completions[i] + "\n```"
    
problems["output"] = completions
problems["raw_output"] = outputs
samples = problems.to_dict(orient="records")
output_filepath = os.path.join(args.save_dir, "samples.jsonl")
write_jsonl(output_filepath, samples)


# eval
import re
import json
from pathlib import Path
from collections import defaultdict
from utils.evaluation_leetcode import evaluate_functional_correctness

version = "20240121-Jul"

DATA_DIR = Path(__file__).parent / "data"

def extract_python_code(generation: str):
    generation = generation.replace("[PYTHON]", '```python').replace("[/PYTHON]", '```')
    if '```python' in generation:
        p_code = re.compile(r'```python\n(.*?)\n```', flags=re.DOTALL)
        code_block = p_code.findall(generation)[0]
        return code_block
    else:
        codelist = re.split("\ndef|\nclass|\nif|\n#|\nprint", generation)
        return codelist[0]
    
def evaluate_main(generation_path: str, result_path: str, temp_dir: str):
    problem_path = args.input_data
    print(problem_path)
    problems = [json.loads(line) for line in open(problem_path, 'r',encoding='utf-8')]

    id2problems = { x['task_id']: x for x in problems }

    results = [json.loads(line) for line in open(generation_path, 'r')]
    for result in results:
        if 'task_id' not in result:
            result['task_id'] = problems[result['index']]['task_id']

        if 'generation' not in result:
            try:
                if 'output' not in result:
                    result['output'] = result['response']
                if result['output'].startswith("\n        "):
                    func_code = extract_python_code(result['prompt_sft']).strip()
                    result['generation'] = func_code + '\n' + result['output']
                else:
                    result['generation'] = extract_python_code(result['output'])
            except:
                result['generation'] = result['output']
    
    with open(result_path, 'w') as fr:
        for result in results:
            fr.write(json.dumps(result) + "\n")

    score = evaluate_functional_correctness(
        input_file=result_path,
        tmp_dir=temp_dir,
        problem_file=problem_path,
        result_path=result_path
    )

    hardness_results = defaultdict(int)
    for result in [json.loads(line) for line in open(result_path, 'r')]:
        problem = id2problems[result['task_id']]

        hardness = problem['meta']['difficulty']
        hardness_results[hardness] += 1
        hardness_results[hardness + "_correct"] += result['passed']

    print("="*100)
    print("Evaluate {} over.".format(generation_path))
    print("Pass@1: {:.3f}".format(score["pass@1"]))
    for key in ["Easy", "Medium", "Hard"]:
        if key.endswith("_correct"):
            continue
        acc = hardness_results[key+"_correct"] / hardness_results[key]
        print("{}: {:.3f}({}/{})".format(key, acc, hardness_results[key+"_correct"],  hardness_results[key]))
    
    score_path = os.path.join(args.save_dir, "result.txt")
    with open(score_path, "w") as f:
        f.write("Pass@1: {:.3f}\n".format(score["pass@1"]))
        for key in ["Easy", "Medium", "Hard"]:
            if key.endswith("_correct"):
                continue
            acc = hardness_results[key+"_correct"] / hardness_results[key]
            f.write("{}: {:.3f}({}/{})\n".format(key, acc, hardness_results[key+"_correct"],  hardness_results[key]))
result_path = output_filepath.replace(".jsonl", "_result.jsonl")
evaluate_main(output_filepath, result_path, temp_dir="output/temp")
