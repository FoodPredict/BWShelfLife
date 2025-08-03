
import pandas as pd
from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and preprocessor
model = joblib.load('model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    moisture = float(request.form['moisture'])
    temperature = float(request.form['temperature'])
    season = request.form['season']
    rh = float(request.form['rh'])
    days_after_milling = float(request.form['days_after_milling'])
    packing = request.form['packing']
    ffa = float(request.form['ffa'])

    # Create a DataFrame from the input data
    input_data = pd.DataFrame([[moisture, temperature, season, rh, days_after_milling, packing, ffa]],
                              columns=['Moisture', 'Temperature', 'Season', 'RH', 'Days_after_milling', 'Packing', 'FFA'])

    # Preprocess the input data
    input_processed = preprocessor.transform(input_data)

    # Make prediction
    prediction = model.predict(input_processed)[0]

    return render_template('index.html', prediction_text=f'Predicted Shelf Life: {prediction:.2f} days')

if __name__ == "__main__":
    # Save the model and preprocessor
    joblib.dump(model, 'model.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')
    app.run(debug=True)
