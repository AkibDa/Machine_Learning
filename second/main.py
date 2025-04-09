import pandas as pd


def get_Disease_Description():
  df_description = pd.read_csv('archive/symptom_Description.csv')
  df_description = df_description.drop_duplicates(subset=['Disease'])
  df_description = df_description.reset_index(drop=True)
  df_description = df_description[df_description['Disease'].notna()]
  df_description = df_description[df_description['Description'].notna()]
  df_description['Disease'] = df_description['Disease'].str.strip()
  df_description['Description'] = df_description['Description'].str.strip()
  df_description = df_description[df_description['Disease'] != '']
  df_description = df_description[df_description['Description'] != '']
  print("Disease Description:")
  print(df_description.head())

def get_Disease_Symptoms():
  df_Disease = pd.read_csv('archive/dataset.csv')
  df_Disease = df_Disease.drop_duplicates(subset=['Disease'])
  df_Disease = df_Disease.reset_index(drop=True)
  df_Disease = df_Disease[df_Disease['Disease'].notna()]
  
def get_Symptoms():
  Symptoms = input("Enter the Symptoms: ")
  Symptoms = Symptoms.split(",")
  Symptoms = [symptom.strip() for symptom in Symptoms]
  Symptoms = [symptom for symptom in Symptoms if symptom != ''] 
    
if __name__ == "__main__":
  get_Symptoms()
  get_Disease_Symptoms()
  get_Disease_Description()
    