import google.generativeai as genai
import pandas as pd
from tqdm import tqdm
import json

def generate_notes(input_csv_path, output_csv_path, api_key):
  
  genai.configure(api_key=api_key)

  # Set up the model
  generation_config = {
    "temperature": 0.9,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 1024,
  }

  safety_settings = [
    {
      "category": "HARM_CATEGORY_HARASSMENT",
      "threshold": "BLOCK_NONE"
    },
    {
      "category": "HARM_CATEGORY_HATE_SPEECH",
      "threshold": "BLOCK_NONE"
    },
    {
      "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
      "threshold": "BLOCK_NONE"
    },
    {
      "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
      "threshold": "BLOCK_NONE"
    }
  ]

  model = genai.GenerativeModel(model_name="gemini-pro",
                                generation_config=generation_config,
                                safety_settings=safety_settings)


  # load data
  mt = pd.read_csv(input_csv_path)
  # change colume name to dialogues
  mt.columns = ['dialogues']

  # start note as a empty pd dataframe with two columns: dialogue and note
  note = []




  for dialogue in tqdm(mt.dialogues):
      for task in ['sub', 'ap']:
          if task == 'sub':
              prompt_parts = [
              "You are a physician writing a clinical note based on a dialogue with the patient. Only write the \"SUBJECTIVE\" part of note, which include the section of [CHIEF COMPLAINT] and [HISTORY OF PRESENT ILLNESS]. Only include information contained in the dialogue.\nFollow the format as the example below:[\"SUBJECTIVE \nCHIEF COMPLAINT \nAnnual health maintenance examination. \nHISTORY OF PRESENT ILLNESS \nThe patient is a pleasant [age]-year-old male who presents for his annual health maintenance examination. He reports no new complaints today. He denies any recent changes in his hearing. He continues to take niacin for his dyslipidemia, and he has had no problems with hemorrhoids in the last 6 months. He also denies any problems with concha bullosa of the left nostril or septal deviation.]\n",
              f"{dialogue}",
              ]
          else:
              prompt_parts = [
              "You are a physician writing a clinical note based on a dialogue with the patient. Only write the \"ASSESSMENT AND PLAN\" section of note. List each medical problem separately. Under each problem, include assessment (such as medical reasoning) and plan (both diagnostic and therapeutic ). At the end, may include a short section on follow up instruction when applicable. Only include information contained in the dialogue.\n Follow the format as the example below:[\"ASSESSMENT AND PLAN: \n1. Possible COPD exacerbation \nAssessment: Increased work of breathing with  wheezing on exam, suggesting COPD exacerbation. He does have frequent COPD exacerbation in the past. Differential diagnosis include pneumonia (though no fever or cough), PE (though no risk factors) or simple viral infection. \nPlan: \n- WIll obtain CXR. \n- Will start duoneb therapy and oral prednisone 30mg Qday. \n2. Hypertension \nAssessment: The patient's blood pressure is well controlled. Plan: \n- Continue lisinopril 20mg Qday. \nFollow-up instructions: \n- return to clinic in 1 week, or sooner of failed to response with current treatment.]\n",
              f"{dialogue}",
              ]
          try:
              response = model.generate_content(prompt_parts)  # Generating content
              print(response.text)  # Printing the generated dialogue
              
              # zip dialogue and respose, make it a list
              note.append([dialogue, response.text, task])  # Appending the generated dialogue to the list

          except Exception as e:
              print(f"An error occurred: {e}")  # Print or handle the error
              note.append([dialogue, "skip", task])


  # save the transcriptions to a file
  note = pd.DataFrame(note)
  # make the first column as dialogue, the second column as note
  note.columns = ['dialogue', 'note', "label"]            

  note.to_csv(output_csv_path, index=False)


if __name__ == '__main__':
    # Read path from the json file
  with open('paths.json', 'r') as f:
      path = json.load(f)
      ACI_Bench_dialogue_csv_path=path['ACI_Bench_dialogue_csv_path']
      ACI_Bench_output_csv_path=path['ACI_Bench_output_csv_path']
      
      Dialogue_G_dialogue_csv_path=path['Dialogue-G-dialogue_csv_path']
      Dialogue_G_output_csv_path=path['Dialogue-G-output_csv_path']
      
      # Here is you api key for Gemini Pro
      api_key=path['api_key']
      
  # Create reference notes for ACI-Bench. 
  generate_notes(ACI_Bench_dialogue_csv_path, ACI_Bench_output_csv_path, api_key)
  
  # Create reference notes for Dialogue-G.
  generate_notes(Dialogue_G_dialogue_csv_path, Dialogue_G_output_csv_path, api_key)    