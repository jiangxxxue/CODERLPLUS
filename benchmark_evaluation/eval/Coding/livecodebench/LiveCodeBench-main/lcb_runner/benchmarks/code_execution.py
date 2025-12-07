import json
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class CodeExecutionProblem:
    question_id: str
    contest_id: str
    contest_date: datetime
    difficulty: str
    function_name: str
    code: str
    input: str
    output: str
    id: str
    problem_id: str
    numsteps: int

    def __post_init__(self):
        pass

    def insert_output(self, output_list: list[str], pred_list: list[str]) -> dict:
        return {
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "difficulty": self.difficulty,
            "function_name": self.function_name,
            "code": self.code,
            "input": self.input,
            "output": self.output,
            "id": self.id,
            "problem_id": self.problem_id,
            "numsteps": self.numsteps,
            "output_list": output_list,
            "pred_list": pred_list,
        }

    def insert_output_evaluation(
        self, output_list: list[str], code_list: list[str], graded_list: list[bool]
    ) -> dict:
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        return output

    def get_evaluation_sample(self) -> dict:
        return {
            "code": self.code,
            "input": self.input,
            "output": self.output,
        }


def load_code_execution_dataset(release_version="release_v1") -> list[CodeExecutionProblem]:
    # Try to load from local parquet files first
    import os
    local_dir = "data/test_data/livecodebench_execution"
    
    if os.path.exists(local_dir):
        try:
            dataset = load_dataset(
                "parquet",
                data_files=f"{local_dir}/data/*.parquet",
            )
            # Get the dataset from the first (and likely only) split
            dataset_split = list(dataset.values())[0] if isinstance(dataset, dict) else dataset
            dataset = [CodeExecutionProblem(**p) for p in dataset_split]  # type: ignore
            print(f"Loaded {len(dataset)} problems from local parquet files")
            return dataset
        except Exception as e:
            print(f"Failed to load from local parquet files: {e}")
            print("Falling back to HuggingFace Hub...")
    
    # Fallback to original HuggingFace Hub loading
    dataset = load_dataset("livecodebench/execution-v2", split="test")
    dataset = [CodeExecutionProblem(**p) for p in dataset]  # type: ignore
    print(f"Loaded {len(dataset)} problems")
    return dataset


if __name__ == "__main__":
    dataset = load_code_execution_dataset()
