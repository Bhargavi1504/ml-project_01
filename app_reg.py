from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model (trained on 5 features)
model = joblib.load(r'C:\Users\BHARGAVI\Downloads\advance_ml\best_model_bagging.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        quantity = float(request.form['Quantity'])
        price = float(request.form['Price'])
        hour = int(request.form['Hour'])
        weekday = int(request.form['Weekday'])
        price_bin = int(request.form['PriceBin'])

        # Pass only 5 features
        input_data = np.array([[quantity, price, hour, weekday, price_bin]])
        prediction = model.predict(input_data)[0]

        return render_template('index.html', prediction=f"₹ {round(prediction, 2)}")

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {e}")



if __name__ == "__main__":
    app.run(debug=True)
