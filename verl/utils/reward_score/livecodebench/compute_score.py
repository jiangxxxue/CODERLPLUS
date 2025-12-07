import json
import re
import traceback
from verl.utils.reward_score.livecodebench import lcb_compute_score, prepare_unit_test_data
import os, pickle
from verl.utils.reward_score.livecodebench.lcb_runner.benchmarks.code_generation import CodeGenerationProblem, load_code_generation_dataset
from verl.utils.reward_score.livecodebench.lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics, check_correctness
from verl.utils.reward_score.livecodebench.lcb_runner.evaluation.pass_k_utils import extract_instance_results
from math_verify import parse, verify
import tempfile
import subprocess
from contextlib import contextmanager
import signal
import ast
import numpy as np


IMPORT_PROMPT='''from typing import *

from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from string import *
from operator import *
from math import *
import math
import datetime
inf = float('inf')

'''


_livecodebench_cache = None

def get_livecodebench_data():
    """获取LiveCodeBench数据集，使用缓存机制避免重复加载"""
    global _livecodebench_cache
    if _livecodebench_cache is None:
        dataset = load_code_generation_dataset(release_version="release_v4")
        _livecodebench_cache = {problem.question_id: problem for problem in dataset}
    return _livecodebench_cache

livecodebench_dir = os.environ.get("LIVECODEBENCH_DATA_PATH", None)


@contextmanager
def timeout_run(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Code execution timeout")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)

def convert_function_to_class_method(raw_code: str, function_name: str) -> str:
    tree = ast.parse(raw_code)
    target_func = None
    new_body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            target_func = node
        else:
            new_body.append(node)
    
    if target_func is None:
        return None

    if not (target_func.args.args and target_func.args.args[0].arg == "self"):
        self_arg = ast.arg(arg="self", annotation=None)
        target_func.args.args.insert(0, self_arg)    
    class_def = ast.ClassDef(
        name="Solution",
        bases=[],
        keywords=[],
        body=[target_func],
        decorator_list=[]
    )
    
    new_body.append(class_def)
    tree.body = new_body
    
    # 使用 ast.unparse 将 AST 转换为代码字符串（Python 3.9+支持）
    new_code = ast.unparse(tree)
    return new_code


def math_verify_reward_function(solution_str, ground_truth):

    ground_truth = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    
    # 0 in case parsing cannot be completed
    try:
        math_verify_parsed = parse(solution_str, parsing_timeout=5)
    except Exception:
        return 0.0
    
    # 0 if parsing is problematic
    if len(math_verify_parsed) < 2:
        return 0.0
    
    # We perform a quick string match first
    if math_verify_parsed[1] in ground_truth:
        return 1.0
    
    # We now fallback to semantic verification
    for gt in ground_truth:
        try:
            if verify(
                parse(f"\\boxed{{{gt}}}", parsing_timeout=5),
                math_verify_parsed,
                timeout_seconds=5,
            ):
                return 1.0
        except Exception:
            continue
    
    # Very unlikely to be correct after the above matches
    return 0.0



def compute_score(completion, test_cases, task=None, timeout=6, is_long_penalty=False, is_binary_reward=True, is_power4_reward=False, return_individual_results=False):
    # solution = completion.split('```python')[-1].split('```')[0]

    if "</think>" in completion:
        solution_str = completion.split("</think>")[1]
    else:
        solution_str = completion
    
    if "question_id" in test_cases:
        try:
            benchmark_dict = get_livecodebench_data()
            question_id = test_cases["question_id"] 
            benchmark_problem = benchmark_dict[question_id]
            
            custom_output = test_cases.copy()
            custom_output["output_list"] = [solution_str]
            
            return lcb_compute_score([custom_output], [benchmark_problem]), None
        except:
            return False, None
    elif 'import_prefix' in test_cases:

        solutions = re.findall(r"```python\n(.*?)```", solution_str, re.DOTALL)
        if len(solutions) == 0:
            return False, None
        try:
            solution = solutions[-1]
            tree = ast.parse(solution)
            solution = test_cases["import_prefix"] + solution

            # 直接使用完整的测试代码，一次性执行所有测试样例
            cur_solution = solution
            cur_solution += "\n" + test_cases['test_code']
            cur_solution += "\ncheck({})".format(test_cases['entry_point'])

            try:
                # 执行代码的逻辑
                success = False
                message = None
                with timeout_run(seconds=2):
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as temp_file:
                        temp_file.write(cur_solution)
                        temp_file.flush()
                        result = subprocess.run(
                            ['python', temp_file.name],
                            capture_output=True,
                            text=True,
                            timeout=timeout
                        )
                        if result.returncode != 0:
                            success = False
                            message = f"Execution error: {result.stderr}"
                        else:
                            success = True
                            message = "Success"
            except TimeoutError:
                success = False
                message = "Code execution timeout"
            except Exception as e:
                success = False
                message = "Execution exception"
                    
            return success, message

        except Exception as e:
            # traceback.print_exc(10)  # 注释掉以减少日志噪音
            return False, f"Code parsing error: {str(e)}"


    elif "inputs" in test_cases:
        try:
            solutions = re.findall(r"```python\n(.*?)```", solution_str, re.DOTALL)
            if len(solutions) == 0 :
                if return_individual_results:
                    return False, None, []
                return False, None
            else:
                solution = solutions[-1]
                try:
                    tree = ast.parse(solution)
                except:
                    if return_individual_results:
                        return False, None, []
                    return False, None

            if isinstance(test_cases, str):
                input_output = json.loads(test_cases)
            elif isinstance(test_cases, dict):
                input_output = test_cases
                test_cases = json.dumps(test_cases)
                
            else:
                assert False
            if "fn_name" in input_output and "class Solution" not in solution:
                solution = convert_function_to_class_method(solution, input_output["fn_name"])
                if not isinstance(solution, str):
                    if return_individual_results:
                        return False, None, []
                    return False, None
            
            metrics = check_correctness(
                {"input_output":test_cases},
                solution,
                debug=False,
                timeout=timeout,
            )

            metrics = list(metrics)
            fixed = []
            for e in metrics[0]:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            metrics[0] = fixed

            # 返回每个测试的结果
            individual_results = []
            if return_individual_results:
                for i, (input_case, output_case, result) in enumerate(zip(input_output["inputs"], input_output["outputs"], fixed)):
                    test_case_content = {"input": input_case, "output": output_case}
                    passed = result in [True, False] and result
                    details = "Passed" if passed else "Failed"
                    
                    individual_results.append({
                        "test_case": test_case_content,
                        "passed": passed,
                        "details": details
                    })

            if is_binary_reward:
                score = sum(metrics[0]) == len(metrics[0])
            else:
                if is_power4_reward:
                    score = (sum((x if x in [False, True] else False) for x in metrics[0])/len(metrics[0]))**4
                else:
                    score = sum((x if x in [False, True] else False) for x in metrics[0])/len(metrics[0])
            
            if return_individual_results:
                return score, metrics, individual_results
            else:
                return score, metrics

        except Exception as e:
            if return_individual_results:
                return False, None, []
            return False, None
    elif "assert_case" in test_cases:
        solutions = re.findall(r"```python\n(.*?)```", solution_str, re.DOTALL)
        if len(solutions) == 0:
            return False, None
        try:
            solution = solutions[-1]
            tree = ast.parse(solution)

            # 直接将所有assert语句一次性添加到解决方案中执行
            test_code = test_cases['assert_case']
            cur_solution = solution
            for assert_stmt in test_code:
                cur_solution += "\n" + assert_stmt
            cur_solution = IMPORT_PROMPT + cur_solution

            try:
                success = False
                message = None
                with timeout_run(seconds=2):
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as temp_file:
                        temp_file.write(cur_solution)
                        temp_file.flush()
                        result = subprocess.run(
                            ['python', temp_file.name],
                            capture_output=True,
                            text=True,
                            timeout=timeout
                        )
                        if result.returncode != 0:
                            success = False
                            message = f"Execution error: {result.stderr}"
                        else:
                            success = True
                            message = "Success"
            except TimeoutError:
                success = False
                message = "Code execution timeout"
            except Exception as e:
                success = False
                message = "Execution exception"
                    
            return success, message

        except Exception as e:
            return False, f"Code parsing error: {str(e)}"

    else:
        try:
            return math_verify_reward_function(solution_str, test_cases), None
        except:
            return False, None
