# app.py
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler from disk
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")

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

        return render_template("prediction.html", prediction_text=prediction_text)
    except Exception as e:
        return render_template("prediction.html", prediction_text="Error: {}".format(e))

if __name__ == "__main__":
    app.run(debug=True)
