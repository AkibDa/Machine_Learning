import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_Disease_Description(disease):
  df_description = pd.read_csv('archive/symptom_Description.csv', index_col=0)
  if disease in df_description.index:
    print("\n🩺 Disease Description:")
    print(df_description.loc[disease]['Description'])
  else:
    print(f"\nNo description found for {disease}.")

def get_Disease(symptoms_list):
  df = pd.read_csv('archive/dataset.csv')
  df.fillna('none', inplace=True)

  symptom_columns = [col for col in df.columns if col.startswith("Symptom_")]
  all_symptoms = sorted(set(
    sym.strip().lower().replace(" ", "_")
    for col in symptom_columns
    for sym in df[col].unique()
    if sym != 'none'
  ))
  def normalize_symptom(sym):
    return sym.strip().lower().replace(" ", "_")

  def encode_symptoms(row):
    row_syms = set(normalize_symptom(row[col]) for col in symptom_columns if row[col] != 'none')
    return [1 if sym in row_syms else 0 for sym in all_symptoms]
    
  X = df.apply(encode_symptoms, axis=1, result_type='expand')
  X.columns = all_symptoms
  y = df['Disease']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  model = DecisionTreeClassifier()
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  print(f"\n✅ Validation Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

  input_vector = [1 if symptom in symptoms_list else 0 for symptom in all_symptoms]

  if sum(input_vector) < 1:
    print("⚠️  Please enter at least one valid symptom.")
    return

  input_df = pd.DataFrame([input_vector], columns=all_symptoms)

  probabilities = model.predict_proba(input_df)[0]
  top_indices = probabilities.argsort()[-3:][::-1]
  top_diseases = model.classes_[top_indices]

  print("\n🔮 Top Predicted Diseases:")
  for i, disease in enumerate(top_diseases, start=1):
    print(f"{i}. {disease} (Confidence: {probabilities[top_indices[i-1]]*100:.2f}%)")

  get_Disease_Description(top_diseases[0])

def get_Symptoms():
  Symptoms = input("\nEnter the Symptoms: ").lower()
  if Symptoms == 'exit':
    print("👋 Exiting the program.")
    return
  Symptoms_list = Symptoms.replace(" ", "_").split(",")
  Symptoms_list = list(set(sym.strip() for sym in Symptoms_list if sym.strip()))
  print("\n📝 Symptoms provided:")
  print(Symptoms_list)
  get_Disease(Symptoms_list)

if __name__ == "__main__":
  print("🩻 Welcome to the Disease Prediction System")
  print("🔹 Please enter the symptoms you are experiencing, separated by commas.")
  print("🔹 Example: itching, skin_rash, stomach_pain")
  print("🔹 Type 'exit' to quit.")
  get_Symptoms()
