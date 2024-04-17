
import json
import pandas as pd
import numpy as np
import datasets
import re

# Note that in our code we saved copies of discharge summaries to different years. This step may not be necessary for your work.

def make_dc_long_and_short(mimic_patients_path, dc_summary_path, dataset_save_path):

    # anchor_years = pd.read_csv("/data/MIMIC-IV/2.0/hosp/patients.csv")
    anchor_years = pd.read_csv(mimic_patients_path)

    # dc_summary = pd.read_csv("/data/hanyinw2/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv")
    dc_summary = pd.read_csv(dc_summary_path)

    # in the dc_summary table, add a column of anchor_year_group based on subject_id in the anchor_years table
    dc_summary['anchor_year_group'] = dc_summary['subject_id'].map(dict(zip(anchor_years.subject_id, anchor_years.anchor_year_group)))

    # find numbers of rows by anchor_year_group
    dc_summary.groupby('anchor_year_group')['subject_id'].count()

    # remove rows where anchor_year_group == 2020 - 2022, as N is too small
    dc_summary = dc_summary[dc_summary['anchor_year_group'] != '2020 - 2022']

    # in anchor_year_group, change " - " to "_"
    dc_summary['anchor_year_group'] = dc_summary['anchor_year_group'].str.replace(" - ", "_")

    ### Regluar expression to extract hospital course
    pattern1  = re.compile("Brief Hospital Course.*\n*((?:\n.*)+?)(Medications on Admission|___  on Admission|___ on Admission)")
    pattern2  = re.compile("Brief Hospital Course.*\n*((?:\n.*)+?)Discharge Medications")
    pattern3  = re.compile("(Brief Hospital Course|rief Hospital Course|HOSPITAL COURSE)\
                        .*\n*((?:\n.*)+?)\
                        (Medications on Admission|Discharge Medications|DISCHARGE MEDICATIONS|DISCHARGE DIAGNOSIS|Discharge Disposition|___ Disposition|CONDITION ON DISCHARGE|DISCHARGE INSTRUCTIONS)")
    pattern4  = re.compile("(Mg-[12].|LACTATE-[12].|Epi-|Gap-|COUNT-|TRF-)___(.*\n*((?:\n.*)+?))(Medications on Admission)")


        # Try more convservaite pattern first, if not working, try less conservative pattern
    def split_note(note):
        if re.search(pattern1, note):
            return re.search(pattern1, note).group(1)
        else:
            if re.search(pattern2, note):
                return re.search(pattern2, note).group(1)
            else:
                if re.search(pattern3, note):
                    return re.search(pattern3, note).group(2)
                else:
                    if re.search(pattern4, note):
                        return re.search(pattern4, note).group(2)
                    else:
                        return None


    for note_type in ["long","short"]:
        if note_type == "long":
            
            dataset_train = []
            dataset_validation = []
            
            for years in dc_summary['anchor_year_group'].unique():
                # select rows where anchor_year_group == component
                dc_summary_years = dc_summary[dc_summary['anchor_year_group'] == years]
                dc_summary_years = dc_summary_years[['hadm_id', 'text']]  
                dc_summary_years.to_csv(f'{dataset_save_path}/discharge_{years}.csv', index=False)
            
       
                dataset = datasets.load_dataset("csv",
                data_files=f"{dataset_save_path}/discharge_{years}.csv",
                split="train")
                dataset = dataset.train_test_split(test_size=0.1, seed=42)
                dataset["validation"] = dataset.pop("test")
                dataset_train.append(dataset['train'])
                dataset_validation.append(dataset['validation'])
            
            dataset_train = datasets.concatenate_datasets(dataset_train)
            dataset_validation = datasets.concatenate_datasets(dataset_validation) 
            
            dataset_long = datasets.DatasetDict({'train': dataset_train, 'validation': dataset_validation})
            dataset_long.save_to_disk(f"{dataset_save_path}/discharge_long_all")
            

        else:        
            dc_summary["text"] = dc_summary["text"].apply(split_note)
            
            # remove rows with missing text
            dc_summary = dc_summary.dropna(subset=['text'])

            # remove rows with duplicated text, only keep the first one. Removed about 1000 rows
            dc_summary = dc_summary.drop_duplicates(subset=['text'], keep='first')


            # Initiate a new dataset
            dataset_train = []
            dataset_validation = []
            
            for years in dc_summary['anchor_year_group'].unique():
                # select rows where anchor_year_group == component
                dc_summary_years = dc_summary[dc_summary['anchor_year_group'] == years]
                dc_summary_years = dc_summary_years[['hadm_id', 'text']]  
                dc_summary_years.to_csv(f'{dataset_save_path}/discharge_short_{years}.csv', index=False)
                
                # load data from csv file

                dataset = datasets.load_dataset("csv",
                data_files=f"{dataset_save_path}/discharge_short_{years}.csv",
                split="train")
                dataset = dataset.train_test_split(test_size=0.1, seed=42)
                dataset["validation"] = dataset.pop("test")
                # dataset.save_to_disk(f"/data/hanyinw2/dc_summary/data/discharge_short_{years}")
                # dataset_train = datasets.concatenate_datasets([dataset_train, dataset['train']])
                dataset_train.append(dataset['train'])
                dataset_validation.append(dataset['validation'])
             
            dataset_train = datasets.concatenate_datasets(dataset_train)
            dataset_validation = datasets.concatenate_datasets(dataset_validation) 
                
            dataset_short = datasets.DatasetDict({'train': dataset_train, 'validation': dataset_validation})
            dataset_short.save_to_disk(f"{dataset_save_path}/discharge_short_all")
            

if __name__ == '__main__':
    # Read path from the json file
    with open('paths.json', 'r') as f:
      path = json.load(f)
      mimic_patients_path=path['mimic_patients_path']
      discharge_summary_path=path['dc_summary_path']
      dataset_save_path=path['dataset_save_path']
      
    make_dc_long_and_short(mimic_patients_path, discharge_summary_path, dataset_save_path)