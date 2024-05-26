# functions/flask_app.py

import json
from flask import Flask, request, jsonify, render_template_string
import os
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler from disk
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template_string(open('templates/index.html').read())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        Avg_Area_Income = float(request.form['Avg_Area_Income'])
        Avg_Area_House_Age = float(request.form['Avg_Area_House_Age'])
        Avg_Area_Number_of_Rooms = float(request.form['Avg_Area_Number_of_Rooms'])
        Avg_Area_Number_of_Bedrooms = float(request.form['Avg_Area_Number_of_Bedrooms'])
        Area_Population = float(request.form['Area_Population'])

        # Apply log transformation
        features = np.array([[Avg_Area_Income, Avg_Area_House_Age, Avg_Area_Number_of_Rooms,
                              Avg_Area_Number_of_Bedrooms, Area_Population]])
        features_log = np.log(features)

        # Standardize features
        features_scaled = scaler.transform(features_log)

        # Make prediction
        prediction = model.predict(features_scaled)

        # Format the prediction result
        prediction_text = "Predicted House Price: ${:.2f}".format(prediction[0])

        return render_template_string(open('templates/prediction.html').read(), prediction_text=prediction_text)
    except Exception as e:
        return jsonify({'error': str(e)})

# Handler function for Netlify
def handler(event, context):
    from flask import Flask
    from werkzeug.middleware.proxy_fix import ProxyFix

    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
    return app(event, context)
