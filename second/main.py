import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def get_Disease_Description(disease):
  df_description = pd.read_csv('archive/symptom_Description.csv', index_col=0)
  print("Disease Description:")
  print(df_description.head())
  
def get_Disease(symptoms_list):
  df_Disease = pd.read_csv('archive/dataset.csv')
  for symptom in symptoms_list:
    if symptom not in df_Disease.columns:
      print(f"Symptom '{symptom}' not found in the dataset.")
      return
  for i in range(len(symptoms_list)):
    df_Disease = df_Disease[df_Disease[f'Symptom_{i}'].str.contains('|'.join(symptoms_list))]
  y = df_Disease['Disease']
  X = df_Disease[symptoms_list]
  
  model = DecisionTreeRegressor(random_state=1)
  model.fit(X, y)
  print("Making predictions for the following symptoms:")
  print(X.head())
  print("The predictions are:")
  predictions = model.predict(X.head())
  print(predictions)
  print("Predicted Disease:")
  print(predictions[0])
  get_Disease_Description(predictions[0])
  
def get_Symptoms():
  Symptoms = input("Enter the Symptoms: ").lower()
  if Symptoms == 'exit':
    print("Exiting the program.")
    return
  Symptoms = Symptoms.replace(" ", "")
  Symptoms_list = Symptoms.split(",")
  Symptoms_list = [symptom.strip() for symptom in Symptoms_list]
  Symptoms_list = list(set(Symptoms_list))
  Symptoms_list = [symptom for symptom in Symptoms_list if symptom != '']
  print("Symptoms:")
  print(Symptoms_list)
  get_Disease(Symptoms_list)
  
    
if __name__ == "__main__":
  print("Welcome to the Disease Prediction System")
  print("Please enter the symptoms you are experiencing, separated by commas.")
  print("For example: itching, skin_rash, stomach_pain")
  print("Please enter 'exit' to quit the program.")
  get_Symptoms()
    