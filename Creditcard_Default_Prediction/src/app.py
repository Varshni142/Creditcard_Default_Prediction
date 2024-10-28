import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import joblib

app = Flask(__name__)
CORS(app)

# Load the pre-trained models and scaler
neural_model = pickle.load(open('neural_credit.pkl', 'rb'))
rf_model = joblib.load('rf_credit.pkl')
dt_model = joblib.load('dt_credit.pkl')
scaler = pickle.load(open('scaler_credit.pkl', 'rb'))

@app.route('/')
def home():
    return "Credit Card Default Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_type = data.get('model_type', 'neural')

        # Remove model_type from data as it's not a feature
        data.pop('model_type', None)

        # Convert the input data to a DataFrame
        input_data = pd.DataFrame([data])

        # Expected feature columns for the prediction
        required_columns = [
            'age', 'income', 'credit_limit', 'outstanding_balance', 
            'payment_history', 'default_history', 'education', 
            'marital_status', 'credit_score', 'num_of_loans', 
            'employment_status', 'residential_status', 'loan_duration',
            'loan_amount', 'payment_ratio', 'delinquent_accounts', 
            'credit_utilization', 'num_credit_cards'
        ]
        input_data = input_data.reindex(columns=required_columns, fill_value=0)

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Predict using the selected model
        if model_type == 'random_forest':
            prediction = rf_model.predict(input_data_scaled)
        elif model_type == 'decision_tree':
            prediction = dt_model.predict(input_data_scaled)
        else:
            prediction = neural_model.predict(input_data_scaled)

        return jsonify({'prediction': int(prediction[0])})  # Return prediction result
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
