#!/bin/sh
export CUDA_VISIBLE_DEVICES=4

# In this scripts, we first perform generation, followed by RLAIF using generated notes.

model_name="Llama-2-13b-chat-hf"
dataset_types="short"
experiments="mixed_precision_scheduler"
model_peft_path="post_SFT/${experiment}/${model_name}/${dataset_type}"
output_path="post_RLHF/${experiment}/${model_name}/${dataset_type}"
note_output_path="post_SFT/${experiment}/${model_name}/${dataset_type}/notes"



python /data/hanyinw2/dc_summary/RLHF/try_DPO_v2.py \
--model_name meta-llama/${model_name} \
--peft_model_0 $model_peft_path \
--output_dir $output_path \
--data_dir $note_output_path \
--num_train_epochs 1 \
--learning_rate 5e-6 \
--wandb_project midLR_newPrompt_DPO \

