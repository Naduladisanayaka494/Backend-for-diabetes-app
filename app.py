# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Welcome to the Diabetes Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()

    # Convert data into an array for prediction
    input_data = np.array([data['Pregnancies'], data['Glucose'], data['BloodPressure'],
                           data['SkinThickness'], data['Insulin'], data['BMI'],
                           data['DiabetesPedigreeFunction'], data['Age']])

    input_data = input_data.reshape(1, -1)

    # Predict using the model
    prediction = model.predict(input_data)[0]

    # Return the result as a JSON response
    return jsonify({'diabetes': bool(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
