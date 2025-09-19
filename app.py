from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# --- Load Model and Preprocessing Artifacts ---
try:
    model = joblib.load("model/dementia_model.joblib")
    scaler = joblib.load("model/scaler.joblib")
    label_encoder = joblib.load("model/label_encoder.joblib")
    selector = joblib.load("model/selector.joblib")
    print("Model and artifacts loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model artifacts: {e}")
    print("Please run train.py to generate the model artifacts.")
    model = None # Set to None to handle gracefully

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

    try:
        # --- Get data from form ---
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        education = float(request.form['education'])
        ses = float(request.form['ses'])
        mmse = float(request.form['mmse'])
        etiv = float(request.form['etiv'])
        nwbv = float(request.form['nwbv'])
        asf = float(request.form['asf'])

        # --- Create input array for prediction ---
        input_data = np.array([[age, sex, education, ses, mmse, etiv, nwbv, asf]])

        # --- Preprocess the input ---
        scaled_data = scaler.transform(input_data)
        selected_data = selector.transform(scaled_data)

        # --- Make prediction ---
        prediction_idx = model.predict(selected_data)[0]
        predicted_class = label_encoder.inverse_transform([prediction_idx])[0]
        
        probabilities = model.predict_proba(selected_data)[0]
        class_labels = label_encoder.classes_
        
        # Create a dictionary of class probabilities
        prob_dict = {label: f"{prob*100:.2f}%" for label, prob in zip(class_labels, probabilities)}

        return jsonify({
            'prediction': predicted_class,
            'probabilities': prob_dict
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 400

if __name__ == '__main__':
    # Before running for the first time, make sure to run train.py
    if not os.path.exists('model/dementia_model.joblib'):
        print("Model not found. Please run train.py first to train and save the model.")
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)
