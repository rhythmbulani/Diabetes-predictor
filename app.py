from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

naive_bayes_model = pickle.load(open("naive_bayes_model.pkl", "rb"))

scaler = pickle.load(open("scaler.pkl", "rb"))

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    age = data['age']
    glucose = data['glucose']
    insulin = data['insulin']
    bmi = data['bmi']

    input_features = np.array([[age, glucose, insulin, bmi]])

    input_scaled = scaler.transform(input_features)

    prediction = naive_bayes_model.predict(input_scaled)

    return jsonify({
        "diabetes_type": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)
