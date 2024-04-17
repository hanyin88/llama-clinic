#!/bin/sh
export CUDA_VISIBLE_DEVICES=4

# In this scripts, we first perform generation, followed by RLAIF using generated notes.

model_name="Llama-2-13b-chat-hf"
dataset_types="short"
experiments="mixed_precision_scheduler"
model_peft_path="post_SFT/${experiment}/${model_name}/${dataset_type}"
note_output_path="post_SFT/${experiment}/${model_name}/${dataset_type}/notes"


python RLHF/inference_sub_RLHF_v2.py \
                        --model_name meta-llama/${model_name} \
                        --peft_model_0 $model \
                        --prompt_file $model_peft_path \
                        --use_fast_kernels False \
                        --output_path $note_output_path \
                        --repetition_penalty 1.2 \
  