# file based on: dc_summary/RLHF/inference_sub_RLHF_v2.py


import fire
import os
import sys
import time

import torch
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM, LlamaConfig

from peft import PeftModel
from llama_recipes.inference.safety_utils import get_safety_checker
from llama_recipes.inference.model_utils import load_model, load_peft_model

import datasets
from tqdm import tqdm

def main(
    model_name: str="meta-llama/Llama-2-13b-chat-hf",
    peft_model_0: str=None,
    peft_model_1: str=None,
    peft_model_2: str=None,
    peft_model_3: str=None,
    peft_model_4: str=None,
    peft_model_5: str=None,
    peft_model_6: str=None,
    # peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =1000, #The maximum numbers of tokens to generate
    prompt_file: str="/srv/local/data/hanyinw2/dc_summary/conversation/conversation_data_sub",
    output_path: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.2, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool=True, # Enable safety check with Salesforce safety flan t5
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    validation: bool=False, # whether to use validation set or not
    use_fast_kernels: bool =True, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):

    if os.path.exists(output_path):
        print(f"Provided output path already exists {output_path}")
        sys.exit(1)

    local_variables = locals()
    for param, value in local_variables.items():
        print(f"{param} = {value}")


    dataset = datasets.load_from_disk(prompt_file)

    if validation:
        dataset = dataset['validation']
    else:
        dataset = dataset['train']

    prompt_sub = ("You are a physician writing a clinical note based on a dialogue with the patient. \
Only write the \"SUBJECTIVE\" part of note. \
Only include information contained in the dialogue."
                  )

    prompt_ap = (
        "You are a physician writing a clinical note based on a dialogue with the patient. \
Only write the \"ASSESSMENT AND PLAN\" part of the notes. \
Only include information contained in the dialogue."
                 )
    
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    def apply_prompt_template(sample):
        diagloge = sample["src"].strip()
        diagloge = tokenizer.encode(diagloge,truncation=True, max_length=2900, add_special_tokens=False)
        diagloge = tokenizer.decode(diagloge, skip_special_tokens=True)

        if sample["label"] == "sub":
            prompt = f"{B_INST}{B_SYS}{prompt_sub.strip()}{E_SYS}###DIALOGUE:\n{diagloge}\n###CLINICAL NOTE-SUBJECTIVE: {E_INST}"
        else:
            prompt = f"{B_INST}{B_SYS}{prompt_ap.strip()}{E_SYS}###DIALOGUE>:\n{diagloge}\n###CLINICAL NOTE-ASSESSMENT AND PLAN: {E_INST}"

        return {
            "prompt": prompt
        }

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
    )



    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)    
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    if peft_model_0 is not None:
        print(f"Loading peft model from {peft_model_0}")
        model = PeftModel.from_pretrained(model, peft_model_0)
        if peft_model_1 is not None:
            model = model.merge_and_unload()
            print(f"Loading peft model from {peft_model_1}")
            model = PeftModel.from_pretrained(model, peft_model_1)
            if peft_model_2 is not None:
                model = model.merge_and_unload()
                print(f"Loading peft model from {peft_model_2}")
                model = PeftModel.from_pretrained(model, peft_model_2)
                
                if peft_model_3 is not None:
                    model = model.merge_and_unload()
                    print(f"Loading peft model from {peft_model_3}")
                    model = PeftModel.from_pretrained(model, peft_model_3)
                    if peft_model_4 is not None:
                        model = model.merge_and_unload()
                        print(f"Loading peft model from {peft_model_4}")
                        model = PeftModel.from_pretrained(model, peft_model_4)
                        if peft_model_5 is not None:
                            model = model.merge_and_unload()
                            print(f"Loading peft model from {peft_model_5}")
                            model = PeftModel.from_pretrained(model, peft_model_5)
                            if peft_model_6 is not None:
                                model = model.merge_and_unload()
                                print(f"Loading peft model from {peft_model_6}")
                                model = PeftModel.from_pretrained(model, peft_model_6)
                            
                     
    model.eval()
    

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    answers = []


    for sample in tqdm(dataset):
        user_prompt = apply_prompt_template(sample)
        user_prompt = tokenizer(user_prompt["prompt"],truncation=True, max_length=3100, return_tensors="pt")

        user_prompt = {k: v.to("cuda") for k, v in user_prompt.items()}

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **user_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs 
            )
        e2e_inference_time = (time.perf_counter()-start)*1000
        print(f"the inference time is {e2e_inference_time} ms")

        prompt_len = user_prompt["input_ids"].shape[1]

        answer = tokenizer.decode(outputs[0][prompt_len :], skip_special_tokens=True)
        
        answers.append(answer)

    # in dataset, change the column name from Note to Note_gold
    dataset = dataset.rename_column("tgt", "note_gold")

    #save the model output as a new column
    dataset = dataset.add_column("note", answers)
    
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    #save the dataset as csv
    dataset = dataset.to_csv(output_path)


if __name__ == "__main__":
    fire.Fire(main)
