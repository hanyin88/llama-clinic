import copy
import datasets
import os


# This is a wrapper function for the Huggingface dataset function. Import + preprocess + mapping

def get_custom_dataset(dataset_config, tokenizer, split):

    train_data_path = dataset_config.path

    file_name = os.path.basename(train_data_path)
    cache_dir = "/data/hanyinw2/dc_summary/data/data_cache"
    cache_file = f"{cache_dir}/{file_name}_{split}_cache"



    print("Loading dataset from {}".format(train_data_path))


    dataset = datasets.load_from_disk(train_data_path)
    dataset = dataset[split]
    

    def preprocess_function(examples):
        tokenized_examples = tokenizer(examples["text"] +  tokenizer.eos_token)
        tokenized_examples["labels"] = copy.deepcopy(tokenized_examples["input_ids"])
        return tokenized_examples

    dataset = dataset.map(preprocess_function, remove_columns=["text", "hadm_id"], cache_file_name=cache_file)
 
    return dataset