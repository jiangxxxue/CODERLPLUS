#!/bin/bash

# IMPORTANT: See and modify  all [] placeholders in this file.   

MODEL_CKPT=[path to model checkpoint]
MODEL_NAME=[model name]


OUTPUT_DIR="benchmark_evaluation/eval/eval_results/$MODEL_NAME"
mkdir -p $OUTPUT_DIR
BASE_DIR=benchmark_evaluation/eval

# Specify the test data set
my_array=(humaneval leetcode livecodebench code_reasoning test_output_generation)

export HF_ENDPOINT=https://hf-mirror.com

if [[ " ${my_array[@]} " =~ " humaneval " ]]; then
    export PATH="usr/miniconda3/envs/eval_bench/bin:$PATH" [set PATH to your conda env]
    echo "which python"
    which python
    echo "running humaneval"
    mkdir -p $OUTPUT_DIR/human_eval_chat
    python $BASE_DIR/Coding/human_eval/evaluate_human_eval.py \
        --model $MODEL_CKPT \
        --data_dir data/test_data/humaneval \
        --save_dir $OUTPUT_DIR/human_eval_chat 
fi


if [[ " ${my_array[@]} " =~ " leetcode " ]]; then
    export PATH="usr/miniconda3/envs/eval_bench/bin:$PATH" [set PATH to your conda env]
    echo "which python"
    which python
    echo "running leetcode"
    mkdir -p $OUTPUT_DIR/leetcode_chat
    python $BASE_DIR/Coding/leetcode/evaluate_leetcode.py \
        --model $MODEL_CKPT \
        --input_data data/test_data/leetcode/leetcode-test.json \
        --save_dir $OUTPUT_DIR/leetcode_chat
fi


if [[ " ${my_array[@]} " =~ " livecodebench " ]]; then
    export PATH="usr/miniconda3/envs/LCB/bin:$PATH" [set PATH to your conda env]
    echo "which python"
    which python
    echo "running livecodebench"
    mkdir -p $OUTPUT_DIR/livecodebench
    cd $BASE_DIR/Coding/livecodebench/LiveCodeBench-main
    python -m lcb_runner.runner.main --model $MODEL_CKPT --scenario codegeneration --evaluate --release_version release_v4 --output_path $OUTPUT_DIR/livecodebench
    nohup python -m lcb_runner.evaluation.compute_scores --eval_all_file $OUTPUT_DIR/livecodebench/result_eval_all.json --start_date 2023-05-01 --end_date 2024-11-01 >$OUTPUT_DIR/livecodebench/lcb_v4.txt 2>&1 &
    cd ../../../
fi


if [[ " ${my_array[@]} " =~ " code_reasoning " ]]; then
    export PATH="usr/miniconda3/envs/LCB/bin:$PATH"
    echo "which python"
    which python
    echo "running livecodebench code execution with custom reasoning prompt"
    mkdir -p $OUTPUT_DIR/livecodebench_code_execution
    python $BASE_DIR/Coding/livecodebench/evaluate_code_execution.py \
        --model $MODEL_CKPT \
        --save_dir $OUTPUT_DIR/livecodebench_code_execution \
        --start_date 2023-05-01 --end_date 2024-11-01 \
        --temperature 0.0 \
        --max_tokens 4096 \
        --n_samples 1
fi


if [[ " ${my_array[@]} " =~ " test_output_generation " ]]; then
    export PATH="usr/miniconda3/envs/LCB/bin:$PATH" [set PATH to your conda env]
    echo "which python"
    which python
    echo "running livecodebench - Test Output Prediction"
    mkdir -p $OUTPUT_DIR/livecodebench_output_prediction
    cd $BASE_DIR/Coding/livecodebench/LiveCodeBench-main
    python -m lcb_runner.runner.main \
        --model $MODEL_CKPT \
        --scenario testoutputprediction \
        --evaluate \
        --output_path $OUTPUT_DIR/livecodebench_output_prediction
    nohup python -m lcb_runner.evaluation.compute_scores --eval_all_file $OUTPUT_DIR/livecodebench_output_prediction/result_eval_all.json --start_date 2023-05-01 --end_date 2024-11-01 >$OUTPUT_DIR/livecodebench_output_prediction/lcb_v4.txt 2>&1 &
    cd ../../../
fi
