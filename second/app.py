from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from groq import Groq
from key import API_KEY

app = Flask(__name__)

def load_data():
  df = pd.read_csv('archive/dataset.csv').fillna('none')
  df_precaution = pd.read_csv('archive/symptom_precaution.csv', index_col=0)
  df_description = pd.read_csv('archive/symptom_Description.csv', index_col=0)
  return df, df_precaution, df_description

def get_disease_info(disease, df_precaution, df_description):
  info = {}
  if disease in df_description.index:
    info['description'] = df_description.loc[disease]['Description']
  if disease in df_precaution.index:
    info['precautions'] = [p for p in df_precaution.loc[disease] if pd.notna(p)]
  return info

def predict_disease(symptoms_list, df, df_precaution, df_description):
    symptom_columns = [col for col in df.columns if col.startswith("Symptom_")]
    all_symptoms = sorted(set(
      sym.strip().lower().replace(" ", "_")
      for col in symptom_columns
      for sym in df[col].unique()
      if sym != 'none'
    ))
    
    input_vector = [1 if symptom in symptoms_list else 0 for symptom in all_symptoms]
    if sum(input_vector) < 1:
      return None, "Please enter at least one valid symptom."
    
    X = df.apply(lambda row: encode_symptoms(row, symptom_columns, all_symptoms), axis=1, result_type='expand')
    X.columns = all_symptoms
    y = df['Disease']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    input_df = pd.DataFrame([input_vector], columns=all_symptoms)
    probabilities = model.predict_proba(input_df)[0]
    top_indices = probabilities.argsort()[-3:][::-1]
    top_diseases = model.classes_[top_indices]
    
    results = []
    for i, disease in enumerate(top_diseases, start=1):
      disease_info = get_disease_info(disease, df_precaution, df_description)
      results.append({
        'rank': i,
        'disease': disease,
        'confidence': f"{probabilities[top_indices[i-1]]*100:.2f}%",
        **disease_info
      })
    
    return results, None

def encode_symptoms(row, symptom_columns, all_symptoms):
  row_syms = set(row[col].strip().lower().replace(" ", "_") for col in symptom_columns if row[col] != 'none')
  return [1 if sym in row_syms else 0 for sym in all_symptoms]

def chatbot_query(prompt):
  try:
    client = Groq(api_key=API_KEY)
    response = client.chat.completions.create(
      model="llama3-70b-8192",
      messages=[
          {"role": "system", "content": "You are a medical assistant. Provide concise, actionable advice."},
          {"role": "user", "content": prompt}
      ],
      temperature=0.7,
      max_tokens=1024
    )
    return response.choices[0].message.content
  except Exception as e:
    return f"Error: {str(e)}"

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/diagnose', methods=['POST'])
def diagnose():
  symptoms = request.form.get('symptoms', '')
  symptoms_list = [s.strip().lower().replace(" ", "_") for s in symptoms.split(",")]
  
  df, df_precaution, df_description = load_data()
  results, error = predict_disease(symptoms_list, df, df_precaution, df_description)
  
  if error:
      return jsonify({'error': error})
  return jsonify({'results': results})

@app.route('/chatbot', methods=['POST'])
def chatbot():
  prompt = request.form.get('prompt', '')
  response = chatbot_query(prompt)
  return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)