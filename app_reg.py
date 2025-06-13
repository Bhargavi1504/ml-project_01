# app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the full pipeline (preprocessing + model)
model_pipeline = joblib.load("final_model_pipeline.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form values
        quantity = float(request.form['Quantity'])
        price = float(request.form['Price'])
        hour = int(request.form['Hour'])
        weekday = int(request.form['Weekday'])
        pricebin = int(request.form['PriceBin'])
        desclength = int(request.form['Desclength'])
        is_return = int(request.form['IsReturn'])

        # Create DataFrame with correct column names
        input_df = pd.DataFrame({
            'Quantity': [quantity],
            'Price': [price],
            'Hour': [hour],
            'Weekday': [weekday],
            'PriceBin': [pricebin],
            'Desclength': [desclength],
            'IsReturn': [is_return]
        })

        # Predict using the pipeline
        prediction = model_pipeline.predict(input_df)[0]
        return render_template('index.html', prediction_text=f"Predicted Total Amount: ₹ {prediction:.2f}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
