# CODERL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment

This repository contains the source code, datasets, and models for the paper **"CODERL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment"**.

The CODERL+ trained models are available for download [here](https://huggingface.co/xueniki/Qwen2.5-Coder-7B-Instruct-CodeRLPLUS/tree/main).

## Training Your Own Model

If you want to train your own model, follow these steps:

### 1. Download Training Data
Download the training dataset [here](https://huggingface.co/datasets/xueniki/data_CodeRLPLUS/tree/main). The training data is derived from the [PRIME](https://arxiv.org/abs/2502.01456) work, with adjusted instructions for code RL training.

### 2. Run Training Script
Execute the training script with the following command:
```bash
bash recipe/coderlplus.sh
```
**Note:** Make sure to modify all `[]` placeholders in the script with your specific paths and configurations.

### 3. Evaluate Your Model
To evaluate the trained model, use:
```bash
bash benchmark_evaluation/eval/run.sh
```

For detailed evaluation instructions, please refer to the [evaluation guide](CODERLPLUS/benchmark_evaluation/eval/README.md).

## 📝 Citation

If you find this work helpful, please cite our paper:
```bibtex
@article{jiang2025coderl+,
  title={CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment},
  author={Jiang, Xue and Dong, Yihong and Liu, Mengyang and Deng, Hongyi and Wang, Tian and Tao, Yongding and Cao, Rongyu and Li, Binhua and Jin, Zhi and Jiao, Wenpin and others},
  journal={arXiv preprint arXiv:2510.18471},
  year={2025}
}
```