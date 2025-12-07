# Evaluation Guide

### Requirements

Before the evaluation, please install the required packages with the following command:

For HumanEval and LeetCode, we use the following virtual environment.

```bash
conda create -n eval_bench python==3.10
conda activate eval_bench
pip install -r requirements_eval.txt
```

For LiveCodeBench, Code_Reasoning, Test_Output_Generation, we use the following virtual environment.

```bash
conda create -n LCB python==3.10
conda activate LCB
pip install -r requirements_LCB.txt
```


### Evaluation

1. **Prepare the model and test data**  
   Ensure your model is in HuggingFace format. If your RL-trained model is not in HF format, you can use `scripts/model_merge.sh` to convert it. Download the test datasets (HumanEval, LeetCode, LiveCodeBench) to the folder `data/test_data/`. Alternatively, we also provide a prepared test_data folder ready for use, available [here](https://huggingface.co/datasets/xueniki/data_CodeRLPLUS/tree/main/test_data).

2. **Configure the evaluation script**  
   Review and modify all `[]` placeholders in `eval/run.sh`. You can specify the datasets to be evaluated by updating the `my_array` variable in the script.

3. **Run the evaluation**  
   Execute the following command (ensure you are in the `eval` directory):
   ```bash
   bash run.sh
   ```

4. **Check the results**  
   The evaluation results will be saved in the `eval_results` directory.
