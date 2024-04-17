import pandas as pd

# Read the data
mt = pd.read_csv('mtsamples.csv')

# print
# Find number of rows with medical_specialty as General Medicine 
mt['medical_specialty'].value_counts()

# How many rows have description with the description value include "SUBJECTIVE" or "History of present illness"
# note contains is different than starts with as contains can use regular express and more liberal
# total 1336 cases
mt['transcription'].str.contains("CHIEF COMPLAINT|SUBJECTIVE|HISTORY OF PRESENT ILLNESS").value_counts()

# drop the rows with description with the description value include "SUBJECTIVE" or "History of present illness"

# select the rows with description with the description value include "SUBJECTIVE" or "History of present illness"
mt_selected = mt[mt['transcription'].str.contains("CHIEF COMPLAINT|SUBJECTIVE|HISTORY OF PRESENT ILLNESS", na=False)]

# filter out those rows with word count less than 20
mt_selected = mt_selected[mt_selected['transcription'].str.count(' ') > 20]

# save the transcriptions to a file
mt_selected['transcription'].to_csv('mtsamples_selected.csv', index=False)