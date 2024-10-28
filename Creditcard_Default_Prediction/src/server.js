const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const axios = require('axios');

const app = express();
app.use(cors());
app.use(express.json());

// Connect to MongoDB
mongoose.connect('mongodb://localhost:27017/credit_card', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

// Define the schema for user data and prediction
const userSchema = new mongoose.Schema({
  age: Number,
  income: Number,
  credit_limit: Number,
  outstanding_balance: Number,
  payment_history: String,
  default_history: String,
  education: Number,
  marital_status: String,
  credit_score: Number,
  num_of_loans: Number,
  employment_status: String,
  residential_status: String,
  loan_duration: Number,
  loan_amount: Number,
  payment_ratio: Number,
  delinquent_accounts: Number,
  credit_utilization: Number,
  num_credit_cards: Number,
  prediction: String, // Store the prediction result
});

const User = mongoose.model('User', userSchema);

// API to save form data and result to MongoDB
app.post('/save', async (req, res) => {
  try {
    const userData = new User(req.body);

    // Call Python API to get the prediction result
    const predictionResult = await makePrediction(userData);
    userData.prediction = predictionResult;

    // Save the user data and prediction result to MongoDB
    await userData.save();
    res.status(200).json({ message: 'Data saved successfully!', prediction: predictionResult });
  } catch (error) {
    console.error('Error saving data:', error);
    res.status(500).json({ error: 'Error saving data!' });
  }
});

// Function to send data to Python Flask API for prediction
const makePrediction = async (data) => {
  try {
    const response = await axios.post('http://localhost:5001/predict', {
      age: data.age,
      income: data.income,
      credit_limit: data.credit_limit,
      outstanding_balance: data.outstanding_balance,
      payment_history: data.payment_history,
      default_history: data.default_history,
      education: data.education,
      marital_status: data.marital_status,
      credit_score: data.credit_score,
      num_of_loans: data.num_of_loans,
      employment_status: data.employment_status,
      residential_status: data.residential_status,
      loan_duration: data.loan_duration,
      loan_amount: data.loan_amount,
      payment_ratio: data.payment_ratio,
      delinquent_accounts: data.delinquent_accounts,
      credit_utilization: data.credit_utilization,
      num_credit_cards: data.num_credit_cards
    });

    return response.data.prediction; // Return the prediction result
  } catch (error) {
    console.error('Error calling prediction API:', error);
    throw error;
  }
};

// Start the server on port 5000
app.listen(5000, () => {
  console.log('Server running on port 5000');
});
