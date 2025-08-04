from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("rf_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = request.form
    input_data = {
        "trans_hour": int(form_data['trans_hour']),
        "trans_day": int(form_data['trans_day']),
        "trans_month": int(form_data['trans_month']),
        "trans_year": int(form_data['trans_year']),
        "trans_amount": float(form_data['trans_amount']),
        "upi_number": form_data['upi_number']  # Add UPI number
    }

    # Create a DataFrame from input
    df = pd.DataFrame([input_data])

    # Predict with the model
    prediction = model.predict(df)[0]
    
    # Convert prediction to label
    result = "Fraud" if prediction == 1 else "Not Fraud"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
