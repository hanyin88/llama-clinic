# LLaMA-Clinic

## This repository contains the code and dataset used for the paper: Adapting Open-Source Large Language Models for Cost-Effective, Expert-Level Clinical Note Generation with On-Policy Reinforcement Learning. 

## Local setup
Install dependencies. We used conda environment.
```
conda env create -f environment.yml
```
Activate conda environment.
```
conda activate LLaMA-Clinic
```

We made minor modificaiton to Meta's llama-receipes to add simple simple functions. Code could be find here: https://github.com/hanyin88/llama-recipes.

## Dataset
Please refer to the data folder. 

## Experiments
Modules and sample scripts for pretraining, SFT, and DPO can be found in the respective folders. 