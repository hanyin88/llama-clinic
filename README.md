# LLaMA-Clinic

## This repository contains the code and dataset used for the paper: Towards Adapting Open-Source Large Language Models for Expert-Level Clinical Note Generation and implementation instructions.

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
Sample scripts for pretraining, SFT, and DPO can be found in the respective folders. 