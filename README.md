
🩺 Heart Failure Risk Predictor

A Flask-based machine learning web application that predicts the risk of heart failure using patient medical data. Built by @reeta-042 to demonstrate applied data science in healthcare and for a devtown capstone project 

🌐 Live Demo
Click here to try the app  
(https://heart-failure-prediction-0020.onrender.com)


🧠 Project Overview

This web app collects patient health indicators through an HTML form and uses a trained machine learning model to assess heart failure risk in real time. The model is hosted using Flask and served with Gunicorn on Render.


📁 Repository Structure

`
heart-failure-predictor/
├── app.py                   # Flask backend logic
├── best_log_model.pkl          # Trained ML model (with Logistic Regression algorithm)
├── scaler.pkl               # Scaler object used for normalization
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
│
├── templates/
│   └── heart2_failured.html      # HTML form for user input
│
├── static/
│   └── medical.jpg    
│
└── notebooks/
    └── EJIABOR_RITA_DEV_TOWN_CAPSTONE_PROJECT.ipynb   # Model training and analysis notebook
`



💡 Features

- Clean, user-friendly web form for patient data entry
- Machine learning-based prediction of heart failure risk
- Dynamic result rendering (risk vs. no risk)
- Medical-themed background for clinical presentation
- Ready for deployment via Render


📦 Requirements

- Flask  
- Scikit-learn  
- NumPy  
- Joblib  
- Gunicorn (for deployment)


📊 Model Information

The model was trained using patient clinical data such as age, ejection fraction, serum creatinine, anaemia, and blood pressure. See notebooks/Ejiabor_Rita_DEV_TOWN_CAPSTONE_PROJECT.ipynb for full preprocessing, feature engineering, and accuracy metrics.


🔐 Disclaimer

This project is for educational and demonstration purposes only . It is not intended for clinical decision-making or diagnostic use.
