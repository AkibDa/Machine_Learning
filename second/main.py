import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def get_Disease_Description(disease):
    df_description = pd.read_csv('archive/symptom_Description.csv', index_col=0)
    if disease in df_description.index:
        print("\nDisease Description:")
        print(df_description.loc[disease]['Description'])
    else:
        print(f"\nNo description found for {disease}.")

def get_Disease(symptoms_list):
    df = pd.read_csv('archive/dataset.csv')
    
    # Fill NaN values with 'none'
    df.fillna('none', inplace=True)

    # Create a symptom list from all symptom columns
    symptom_columns = [col for col in df.columns if col.startswith("Symptom_")]
    
    # Encode symptoms into 0/1 (binary vector)
    all_symptoms = set()
    for col in symptom_columns:
        all_symptoms.update(df[col].unique())
    all_symptoms = list(all_symptoms - {'none'})  # remove placeholder
    
    def encode_symptoms(row):
        symptoms = set(row[col] for col in symptom_columns if row[col] != 'none')
        return [1 if symptom in symptoms else 0 for symptom in all_symptoms]
    
    X = df.apply(encode_symptoms, axis=1, result_type='expand')
    X.columns = all_symptoms
    
    y = df['Disease']
    
    # Train the model
    model = DecisionTreeClassifier()
    model.fit(X, y)

    # Encode input symptoms
    input_vector = [1 if symptom in symptoms_list else 0 for symptom in all_symptoms]
    
    # Predict
    prediction = model.predict([input_vector])[0]
    print("\nPredicted Disease:", prediction)
    
    get_Disease_Description(prediction)

def get_Symptoms():
    Symptoms = input("\nEnter the Symptoms: ").lower()
    if Symptoms == 'exit':
        print("Exiting the program.")
        return
    Symptoms = Symptoms.replace(" ", "")
    Symptoms_list = Symptoms.split(",")
    Symptoms_list = list(set([symptom.strip() for symptom in Symptoms_list if symptom.strip() != '']))
    print("\nSymptoms provided:")
    print(Symptoms_list)
    get_Disease(Symptoms_list)

if __name__ == "__main__":
    print("Welcome to the Disease Prediction System")
    print("Please enter the symptoms you are experiencing, separated by commas.")
    print("For example: itching, skin_rash, stomach_pain")
    print("Type 'exit' to quit.")
    get_Symptoms()
