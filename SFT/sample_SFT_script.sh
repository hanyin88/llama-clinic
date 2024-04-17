#!/bin/sh
export CUDA_VISIBLE_DEVICES=2,3,5,6

model_names="Llama-2-13b-hf"
dataset_types="short"
experiments="mixed_precision_scheduler"
model_peft_path="post_cp/${experiment}/${model_name}/${dataset_type}"
output_path="post_SFT/${experiment}/${model_name}/${dataset_type}"


torchrun --nnodes 1 \
--nproc_per_node 4  \
/data/hanyinw2/llama-recipes/examples/finetuning.py \
--dataset custom_dataset \
--custom_dataset.file /data/hanyinw2/llama_clinic/SFT/conversation_dataset_SFT.py \
--custom_dataset.path /data/hanyinw2/dc_summary/conversation/data/conversation_data_aug_sft_clean_newPrompt \
--lr 2e-5 \
--batching_strategy padding \
--weight_decay 0.1 \
--enable_fsdp \
--model_name meta-llama/${model_name} \
--use_peft \
--num_epochs 3 \
--peft_method lora \
--target_modules "[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\"]" \
--output_dir $output_path \
--resume_from_checkpoint $model_peft_path \
--use_fast_kernels False\
--use_wandb \
--wandb_project conversation_sft \



            
