#!/bin/sh
export CUDA_VISIBLE_DEVICES=4,5,6,7

model="Llama-2-13b-hf"
dataset_type="long"
output_path="post_CP/${experiment}/${model_name}/${dataset_type}"

torchrun --nnodes 1 --nproc_per_node 4 /data/hanyinw2/llama-recipes/examples/finetuning.py --dataset custom_dataset --custom_dataset.file pretraining/discharge_dataset.py --custom_dataset.path /data/hanyinw2/dc_summary/data/discharge_${dataset_type}_all --lr 3e-4 --weight_decay 0.1 --enable_fsdp --model_name meta-llama/${model} --use_peft --num_epochs 1 --peft_method lora --target_modules "[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\"]" --output_dir $output_path --use_fast_kernels --use_wandb True --run_name ${model}_${dataset_type}_mixed_precision_scheduler --use_cosine_scheduler True 
        
