#!/bin/bash

# IMPORTANT: See and modify  all [] placeholders in this file.   

hf_model_path=[raw hf_model_path]
target_model_dir=[the model you want to change to hf format]

python model_merger.py \
    --backend fsdp \
    --tie-word-embedding \
    --hf_model_path ${hf_model_path} \
    --local_dir ${target_model_dir} \
    --target_dir ${target_model_dir}/huggingface/ \

cp ${hf_model_path}/added_tokens.json ${target_model_dir}/huggingface/added_tokens.json
cp ${hf_model_path}/merges.txt ${target_model_dir}/huggingface/merges.txt
cp ${hf_model_path}/special_tokens_map.json ${target_model_dir}/huggingface/special_tokens_map.json
cp ${hf_model_path}/tokenizer_config.json ${target_model_dir}/huggingface/tokenizer_config.json
cp ${hf_model_path}/tokenizer.json ${target_model_dir}/huggingface/tokenizer.json
cp ${hf_model_path}/vocab.json ${target_model_dir}/huggingface/vocab.json