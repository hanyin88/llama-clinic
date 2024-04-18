# file based on: dc_summary/RLHF/try_DPO_v2.py

import fire
import os
import sys
import time

import torch
from transformers import LlamaTokenizer, TrainingArguments
from transformers import LlamaForCausalLM, LlamaConfig
from datasets import load_dataset

from peft import PeftConfig, PeftModel, AutoPeftModelForCausalLM,LoraConfig, prepare_model_for_int8_training
from llama_recipes.inference.safety_utils import get_safety_checker
from llama_recipes.inference.model_utils import load_model, load_peft_model



from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, DPOTrainer

# write a function for all the following steps
# 1. load a pretrained model
# 2. load dataset from csv
# 3. create a trainer
# 4. train the model
# 5. save the model

def DPO(
        model_name: str="meta-llama/Llama-2-13b-chat-hf",
        peft_model_0: str=None,
        peft_model_1: str=None,
        peft_model_2: str=None,
        peft_model_3: str=None,
        peft_model_4: str=None,
        peft_model_5: str=None,
        peft_model_6: str=None,
        ref_model: str=None,
        output_dir: str=None,
        data_dir: str=None,  
        quantization: bool=False,
        batch_size: int=1,
        gradient_accumulation_steps: int=8,
        num_train_epochs: int=3,
        optim: str="paged_adamw_32bit",
        learning_rate: float=2e-5,
        max_length: int=4096,
        max_prompt_length: int=3000,
        PEFT_DPO: bool=True,
        wandb_run_name: str=None,
        wandb_project: str=None,
        ):


    if os.path.exists(output_dir + "/config.json"):
        print("config.json exists, exit")
        sys.exit(0)
        
    if os.path.exists(output_dir + "/adapter_config.json"):
        print("adapter_config.json exists, exit")
        sys.exit(0)

    # 1. load a pretrained model
    model =LlamaForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=quantization,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                )

    # Note if do quantization, need to use prepare_model_for_int8_training(model) first before load with peft
    # Turns out this  won't work, as DPO trainer always request model merge, and simply  Cannot merge LORA layers when the model is loaded in 8-bit mode
    # https://github.com/artidoro/qlora/issues/29
    if quantization:
        model = prepare_model_for_int8_training(model)


    if peft_model_0 is not None:
        print(f"Loading peft model from {peft_model_0}")
        model = PeftModel.from_pretrained(model, peft_model_0, is_trainable=True)
        if peft_model_1 is not None:
            model = model.merge_and_unload()
            print(f"Loading peft model from {peft_model_1}")
            model = PeftModel.from_pretrained(model, peft_model_1, is_trainable=True)
            if peft_model_2 is not None:
                model = model.merge_and_unload()
                print(f"Loading peft model from {peft_model_2}")
                model = PeftModel.from_pretrained(model, peft_model_2, is_trainable=True)
                if peft_model_3 is not None:
                    model = model.merge_and_unload()
                    print(f"Loading peft model from {peft_model_3}")
                    model = PeftModel.from_pretrained(model, peft_model_3, is_trainable=True)
                    if peft_model_4 is not None:
                        model = model.merge_and_unload()
                        print(f"Loading peft model from {peft_model_4}")
                        model = PeftModel.from_pretrained(model, peft_model_4, is_trainable=True)
                        if peft_model_5 is not None:
                            model = model.merge_and_unload()
                            print(f"Loading peft model from {peft_model_5}")
                            model = PeftModel.from_pretrained(model, peft_model_5, is_trainable=True)
                            if peft_model_6 is not None:
                                model = model.merge_and_unload()
                                print(f"Loading peft model from {peft_model_6}")
                                model = PeftModel.from_pretrained(model, peft_model_6, is_trainable=True)  

    if ref_model is not None:
        # if ref_model starts with "meta-llama":
        if ref_model.startswith("meta-llama"):
            ref_model = LlamaForCausalLM.from_pretrained(
                    ref_model,
                    load_in_8bit=quantization,
                    device_map="auto",
                    torch_dtype=torch.bfloat16
                    )
            ref_model.eval()
        else:
            base_model = LlamaForCausalLM.from_pretrained(
                        model_name,
                        load_in_8bit=quantization,
                        device_map="auto",
                        torch_dtype=torch.bfloat16
                        )
            if quantization:
                base_model = prepare_model_for_int8_training(base_model)

            ref_model = PeftModel.from_pretrained(base_model, ref_model)

    if PEFT_DPO:
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        # need to do this as in the dpo_trainer.py, it only merges if  model is a peft model and we have a peft_config, we merge and unload it first
        peft_config = None
        if peft_model_0 is not None:
            model = model.merge_and_unload()
        if isinstance(ref_model, PeftModel):
            ref_model = ref_model.merge_and_unload()
            ref_model.eval()

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token



    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing =True,
        num_train_epochs=num_train_epochs, 
        learning_rate=learning_rate,
        bf16=True,
        save_total_limit=1,
        logging_steps=10,
        output_dir=output_dir,
        optim=optim,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        run_name= wandb_run_name,
        )


    prompt_sub = ("You are a physician writing a clinical note based on a dialogue with the patient. Only write the \"SUBJECTIVE\" part of note, which include the section of [CHIEF COMPLAINT] and [HISTORY OF PRESENT ILLNESS]. Only include information contained in the dialogue.\n"
                    )

    prompt_ap = ("You are a physician writing a clinical note based on a dialogue with a patient. Only write the \"ASSESSMENT AND PLAN\" section of note. List each medical problem separately. Explain medical reasoning, diagnostic and therapeutic plans for each problem. At the end, may include a short section on follow up instruction when applicable. Only include information contained in the dialogue.\n"
                    )
        
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    def return_prompt_and_responses(samples):
        diagloges = samples["src"]
        # temporarily fix, but truncate dialogue part to the first 2900 tokens (leave room for the prompt)
        diagloges = [tokenizer.encode(diagloge, truncation=True, max_length=2900, add_special_tokens=False) for diagloge in diagloges]
        diagloges = [tokenizer.decode(diagloge, skip_special_tokens=True) for diagloge in diagloges]

        if samples["label"] == "sub":
            prompt = [B_INST + B_SYS + prompt_sub.strip() + 
                    E_SYS + "###DIALOGUE:\n" + 
                    diagloge + "\n###CLINICAL NOTE-SUBJECTIVE:" + 
                    E_INST for diagloge in diagloges]
        else:
            prompt = [B_INST + B_SYS + prompt_ap.strip() + 
                    E_SYS + "###DIALOGUE:\n" + 
                    diagloge + "\n###CLINICAL NOTE-ASSESSMENT AND PLAN:" 
                    + E_INST for diagloge in diagloges]

        return {
            "prompt": prompt,
            "chosen": samples["note_gold"],   # rated better than k
            # "rejected": samples["note"] if samples["note"] not None else ""
            "rejected": samples["note"]    # rated worse than j      
        }

    # 2. load dataset from csv

    dataset = load_dataset("csv",
        data_files=data_dir,
        split="train",
    )
    original_columns = dataset.column_names

    # filter out the samples with empty note
    dataset = dataset.filter(lambda x: x["note"] is not None)

    dataset = dataset.map(
        return_prompt_and_responses,
        batched=True,
        remove_columns=original_columns
    )


    print(training_args)
    if wandb_project is not None:
        os.environ["WANDB_PROJECT"] = wandb_project

    dpo_trainer = DPOTrainer(
        model,
        # model_ref,
        ref_model=ref_model,
        args=training_args,
        beta=0.1,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_length=max_length,
        max_prompt_length=max_prompt_length
    )

    dpo_trainer.train()
    dpo_trainer.save_model()
    dpo_trainer.save_state()

if __name__ == "__main__":
    fire.Fire(DPO)
 