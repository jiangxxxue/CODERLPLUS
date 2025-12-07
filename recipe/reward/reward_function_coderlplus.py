import json
import re
import logging
from typing import Dict, Any, Optional

from verl.utils.reward_score.livecodebench.compute_score import compute_score as lcb_compute_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_boxed_content(text: str) -> Optional[str]:
    """
    Extract content from the LAST \boxed{} format only.
    
    Args:
        text (str): Text potentially containing \boxed{content}
        
    Returns:
        Optional[str]: Extracted content from the last \boxed{} or None if not found
    """
    if not text:
        return None
        
    patterns = [
        r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',  # Handles nested braces
        r'\\boxed\{([^}]+)\}',  # Simple pattern fallback
        r'boxed\{([^}]+)\}',    # Without leading backslash
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
    
    return None


def normalize_for_comparison(text: str) -> str:
    """
    Normalize text for comparison - handle numeric equivalence and newlines.
    
    Args:
        text (str): Text to normalize
        
    Returns:
        str: Normalized text
    """
    if not text:
        return ""
    
    text = text.strip()
    
    text = text.replace('\\n', '\n')
    
    try:
        import re
        # Use regex to detect valid numeric formats
        # Matches: integers, floats, negative numbers, scientific notation
        numeric_pattern = r'^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$'
        
        if re.match(numeric_pattern, text):
            num = float(text)
            # If it's a whole number, convert to int string
            if num.is_integer():
                return str(int(num))
            else:
                # Use consistent formatting to avoid floating point precision issues
                return f"{num:.10g}"  # Removes trailing zeros and uses scientific notation when needed
    except (ValueError, AttributeError, OverflowError):
        pass
    
    return text


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Dict[str, Any] = None, **kwargs) -> float:
    """
    Compute reward score with data_source-based routing - verl framework compatible.
    
    This function routes to the appropriate scoring method based on data_source:
    - exec_semantics_align: Uses \boxed{} extraction and exact matching
    - codecontests, apps, taco, codeforces: Uses default verl reward scoring (code execution)
    
    Args:
        data_source: Source of the data (e.g., 'exec_semantics_align', 'codecontests')
        solution_str: The model's generated response
        ground_truth: Ground truth information (JSON string)
        extra_info: Additional information
        
    Returns:
        float: Reward score (1.0 for correct, 0.0 for incorrect)
    """
    if extra_info is None:
        extra_info = {}
        
    if "exec_semantics_align" in data_source:
        return compute_exec_semantics_align_score(data_source, solution_str, ground_truth, extra_info)
    elif "code_execution" in data_source:
        return compute_code_reasoning_score(data_source, solution_str, ground_truth, extra_info)
    else:
        score, _  = lcb_compute_score(
            completion=solution_str,
            test_cases=ground_truth,
            is_binary_reward=True,
            is_power4_reward=False
        )

        return float(score) if score is not False else 0.0
        

def compute_code_reasoning_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Dict[str, Any] = None) -> float:
    """
    Compute reward score specifically for execution semantics alignment tasks using \boxed{} extraction.
    
    Args:
        data_source: Source of the data (should contain 'code_reasoning')
        solution_str: The model's generated response
        ground_truth: Ground truth information (JSON string)
        extra_info: Additional information
        
    Returns:
        float: Reward score (1.0 for correct, 0.0 for incorrect)
    """
    if extra_info is None:
        extra_info = {}
        
    try:
        logger.debug(f"Computing execution semantics alignment reward for data_source: {data_source}")
        logger.debug(f"Solution length: {len(solution_str)}")
        logger.debug(f"Ground truth: {str(ground_truth)[:100]}...")
        
        if isinstance(ground_truth, str):
            try:
                gt_data = json.loads(ground_truth)
            except json.JSONDecodeError:
                gt_data = {'expected_output': ground_truth}
        else:
            gt_data = ground_truth

        expected_output = gt_data.get('expected_output', '')
        test_input = gt_data.get('input', '')
        
        logger.debug(f"Expected output: {repr(expected_output)}")
        logger.debug(f"Test input: {repr(test_input)}")
        
        predicted_output = extract_boxed_content(solution_str)
        
        if predicted_output is None:
            logger.warning(f"No \\boxed{{}} content found in execution semantics alignment solution: {solution_str[:100]}...")
            return 0.0
            
        normalized_predicted = normalize_for_comparison(predicted_output)
        normalized_expected = normalize_for_comparison(expected_output)
        
        is_correct = normalized_predicted == normalized_expected
        score = 1.0 if is_correct else 0.0
        
        logger.info(f"Code reasoning score: {score} ({'CORRECT' if is_correct else 'INCORRECT'})")
        logger.info(f"Predicted: {repr(predicted_output)} -> normalized: {repr(normalized_predicted)}")
        logger.info(f"Expected: {repr(expected_output)} -> normalized: {repr(normalized_expected)}")
        
        return score
        
    except Exception as e:
        logger.error(f"Error in compute_code_reasoning_score: {e}")
        logger.error(f"Ground truth: {repr(ground_truth)}")
        logger.error(f"Solution: {solution_str[:200]}...")
        return 0.0


def compute_exec_semantics_align_score(data_source: str, solution_str: str, ground_truth: str, extra_info: Dict[str, Any] = None) -> float:
    """
    Compute reward score specifically for execution semantics alignment tasks.
    
    Args:
        data_source: Source of the data (should contain 'exec_semantics_align')
        solution_str: The model's generated response
        ground_truth: Ground truth information (JSON string)
        extra_info: Additional information
        
    Returns:
        float: Reward score (0.0-1.0)
    """
    if extra_info is None:
        extra_info = {}
        
    try:
        logger.debug(f"Computing execution semantics alignment reward for data_source: {data_source}")
        logger.debug(f"Solution length: {len(solution_str)}")
        logger.debug(f"Ground truth: {str(ground_truth)[:100]}...")
        
        if isinstance(ground_truth, str):
            try:
                gt_data = json.loads(ground_truth)
            except json.JSONDecodeError:
                gt_data = {'expected_output': ground_truth}
        else:
            gt_data = ground_truth
            
        expected_output = gt_data.get('expected_output', '')
        expected_variables = gt_data.get('expected_local_variable', {})
        
        logger.debug(f"Expected output: {repr(expected_output)}")
        logger.debug(f"Expected variables: {repr(expected_variables)}")
        
        lines = solution_str.strip().split('\n')
        if not lines:
            logger.warning("Empty solution string")
            return 0.0
            
        last_line = lines[-1].strip()
        
        try:
            predicted_json = json.loads(last_line)
        except json.JSONDecodeError:
            logger.warning(f"Last line is not valid JSON: {last_line[:100]}...")
            return 0.0
        
        if 'final_output' not in predicted_json or 'variables' not in predicted_json:
            logger.warning(f"JSON missing required keys. Found keys: {predicted_json.keys()}")
            return 0.0
            
        format_score = 0.1
        logger.info("Format check passed: +0.1 score")
        
        predicted_output = predicted_json['final_output']
        predicted_variables = predicted_json.get('variables', {})
        
        total_items = 1 + len(expected_variables)  # 1 for final_output
        correct_items = 0
        
        if check_value_equality(predicted_output, expected_output):
            correct_items += 1
            logger.info(f"final_output correct: {repr(predicted_output)}")
        else:
            logger.info(f"final_output incorrect: predicted={repr(predicted_output)}, expected={repr(expected_output)}")
        
        for var_name, expected_value in expected_variables.items():
            if var_name in predicted_variables:
                predicted_value = predicted_variables[var_name]
                if check_value_equality(predicted_value, expected_value):
                    correct_items += 1
                    logger.info(f"Variable '{var_name}' correct: {repr(predicted_value)}")
                else:
                    logger.info(f"Variable '{var_name}' incorrect: predicted={repr(predicted_value)}, expected={repr(expected_value)}")
            else:
                logger.info(f"Variable '{var_name}' missing in prediction")
        
        value_score = (correct_items / total_items) * 0.9
        
        final_score = format_score + value_score
        
        logger.info(f"Scoring summary:")
        logger.info(f"  - Format score: {format_score}")
        logger.info(f"  - Correct items: {correct_items}/{total_items}")
        logger.info(f"  - Value score: {value_score:.3f}")
        logger.info(f"  - Final score: {final_score:.3f}")
        
        return final_score
        
    except Exception as e:
        logger.error(f"Error in compute_exec_semantics_align_score: {e}")
        logger.error(f"Ground truth: {repr(ground_truth)}")
        logger.error(f"Solution: {solution_str[:200]}...")
        return 0.0


def check_value_equality(predicted, expected):
    """
    Check if two values are equal with special handling for common equivalences.
    
    Handles:
    - Numeric equivalence (1.0 == 1)
    - Boolean/string equivalence (True == "true", False == "false")
    - None/null equivalence (None == "null", None == "none")
    - String comparison (case-sensitive for general strings)
    - List/dict deep comparison
    """
    if predicted == expected:
        return True
    
    pred_str = str(predicted).strip()
    exp_str = str(expected).strip()
    
    none_values = {'None', 'none', 'null', 'Null', 'NULL'}
    if pred_str in none_values and exp_str in none_values:
        return True
    
    true_values = {'True', 'true', 'TRUE', '1'}
    false_values = {'False', 'false', 'FALSE', '0'}
    
    if pred_str in true_values and exp_str in true_values:
        return True
    if pred_str in false_values and exp_str in false_values:
        return True
    
    try:
        pred_num = float(predicted) if not isinstance(predicted, bool) else None
        exp_num = float(expected) if not isinstance(expected, bool) else None
        
        if pred_num is not None and exp_num is not None:
            if abs(pred_num - exp_num) < 1e-9:
                return True
    except (ValueError, TypeError):
        pass

    if isinstance(predicted, list) and isinstance(expected, list):
        if len(predicted) != len(expected):
            return False
        return all(check_value_equality(p, e) for p, e in zip(predicted, expected))
    
    if isinstance(predicted, dict) and isinstance(expected, dict):
        if set(predicted.keys()) != set(expected.keys()):
            return False
        return all(check_value_equality(predicted[k], expected[k]) for k in predicted.keys())
    
    return pred_str == exp_str


def normalize_for_comparison(value):
    return str(value).strip()