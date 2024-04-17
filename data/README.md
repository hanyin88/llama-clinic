# Dataset

### MIMIC-IV Discharge Summaries
1. You must have obtained access to MIMIC-IV database: https://physionet.org/content/mimiciv/. 
2. You can run mimic_preprocessing.py to generate Discharge-long and Discharge-short dataset used in the paper.  You need to provide fields for mimic_patients_path (path to the patients.csv from MIMIC-IV 2.0), dc_summary_path (path to discharge.csv from MIMIC-IV-note 2.2) and dataset_save_path in paths.json.


### Creation of Reference Notes for ACI-Bench and Dialogue G
We included code and prompts used to create reference notes in ACI-Bench and Dialogue-G in gen_note_new_format.py. You need to provide fields for ACI_Bench_dialogue_csv_path, ACI_Bench_output_csv_path, Dialogue_G_dialogue_csv_path, Dialogue_G_output_csv_path and api_key in paths.json to generate reference note for ACI-Bench and Dialogue-G. 

The input dialogue file should contain a single column of dialogue. The output file will include 3 columns of "dialogue", "note" and "label". Note for each dialogue, there will be two separate reference notes for "Subject" and "Assessment and Plan". 

### ACI-Bench
1. The ACI-Bench folder contains ACI-Bench dialogues with our newly created reference notes as discussed above.
2. Original ACI-Bench dialogue could be found on https://github.com/wyim/aci-bench.

### Dialogue-G
1. The Dialogue-G folder contains our newly created synthetic Dialogue-G dataset.
2. Original MT-Samples Data could be found on https://mtsamples.com/.  We downloaded csv file from this Kaggle competition: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions.
3. We first utilized Gemini Pro to transform these notes into dialogues. Subsequently, we used these dialogues as inputs for Gemini Pro once again, this time to generate clinical notes based on ``best practice" format described above. 

### SFT and RLAIF Data Split
1. We combined the training subsets from ACI-BENCH (dialogue n = 67) and Dialogue-G (dialogue n = 1291), then split this data equally for SFT and RLAIF, stratified by data source.
2. The dataset used for SFT and RLAIF are saved as SFT_data and RLAIF_data resepctively. Theese two dataset could be loaded with huggingface's datasets such as:
```
RLAIF_dataset = datasets.load_from_disk("/dc_summary_pub/data/RLAIF_data")
```

### Model outputs with RLAIF
1. We included the training data for each round of modified-dDPO in the folder named Model_outputs_modified_dDPO. In each training cycle, we begin by sampling a response from the latest model check point, and this sampled response is then designated as the rejected response, while a reference response from Gemini Pro (the teacher model) is considered the preferred outcome. 
2. The subfolder 13b_r1 contains the training data used for round 1 of the modified dDPO, and 13b_r2 contains the training data for round 2, and so on. The model's name is reflected in the CSV file name. Files ending in 'baseSFT' represent models that did not undergo continued pretraining.

### RLHF Dataset
1. The RLHF_data folder contains the physician labelled data used for the 2 rounds of RLHF/DPO. 
2. For RLHF, we utilized dialogues from the training, task3, and taskC subsets of ACI-BENCH (dialogue n = 147) as input prompts, and sampled from our model's outputs for physician preference labeling.