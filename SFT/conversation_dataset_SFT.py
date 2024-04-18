import copy
import datasets


def get_custom_dataset(dataset_config, tokenizer, split):
    train_data_path = dataset_config.path

    print("Loading dataset from {}".format(train_data_path))


    dataset = datasets.load_from_disk(train_data_path)
    dataset = dataset[split]

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    
    prompt_sub = ("You are a physician writing a clinical note based on a dialogue with the patient. \
Only write the \"SUBJECTIVE\" part of note. \
Only include information contained in the dialogue."
                  )

    prompt_ap = (
        "You are a physician writing a clinical note based on a dialogue with the patient. \
Only write the \"ASSESSMENT AND PLAN\" part of the notes. \
Only include information contained in the dialogue."
                 )


# following: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    def apply_prompt_template(sample):
        diagloge = sample["src"].strip()
        if sample["label"] == "sub":
            prompt = f"{B_INST}{B_SYS}{prompt_sub.strip()}{E_SYS}###DIALOGUE:\n{diagloge}\n###CLINICAL NOTE-SUBJECTIVE: {E_INST}"
        else:
            prompt = f"{B_INST}{B_SYS}{prompt_ap.strip()}{E_SYS}###DIALOGUE>:\n{diagloge}\n###CLINICAL NOTE-ASSESSMENT AND PLAN: {E_INST}"

        return {
            "prompt": prompt,
            "summary": sample["tgt"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):

        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False, max_length=3000, truncation=True)
        summary = tokenizer.encode(sample["summary"], add_special_tokens=False, max_length=1000, truncation=True)
        summary += tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
 
    return dataset
