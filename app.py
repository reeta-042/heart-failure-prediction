from flask import Flask, render_template, request
import joblib
import numpy as np

# Load model and scaler
model = joblib.load(open('best_log_model.pkl', 'rb'))
scaler = joblib.load(open('scaler.pkl', 'rb'))  

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('heart2_failure.html')

@app.route('/predict', methods=['POST'])
def predict_risk():
    # Collect input from form
    age = float(request.form.get('age'))
    anaemia = int(request.form.get('anaemia'))
    creatinine_phosphokinase = int(request.form.get('creatinine_phosphokinase'))
    diabetes = int(request.form.get('diabetes'))
    ejection_fraction = int(request.form.get('ejection_fraction'))
    high_blood_pressure = int(request.form.get('high_blood_pressure'))
    platelets = float(request.form.get('platelets'))
    serum_creatinine = float(request.form.get('serum_creatinine'))
    serum_sodium = float(request.form.get('serum_sodium'))
    sex = int(request.form.get('sex'))
    smoking = int(request.form.get('smoking'))
    time = int(request.form.get('time'))

    # Prepare input
    features = [age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                high_blood_pressure, platelets, serum_creatinine, serum_sodium,
                sex, smoking, time]
    input_array = np.array(features).reshape(1, -1)

    # Scale input with scaler
    input_scaled = scaler.transform(input_array)  

    # Make prediction
    prediction = model.predict(input_scaled)

    # Interpret result
    if prediction[0] == 1:
        result = "Patient is AT risk of heart failure. Begin treatment in earnest."
    else:
        result = "Patient is NOT at risk of heart failure."

    # Send result to template
    return render_template('heart2_failure.html', result=result)
