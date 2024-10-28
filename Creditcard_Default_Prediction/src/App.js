import React, { useState, useEffect } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

function App() {
  const [step, setStep] = useState(1);
  const [animationClass, setAnimationClass] = useState('');
  const [formData, setFormData] = useState({
    age: '',
    income: '',
    credit_limit: '',
    outstanding_balance: '',
    payment_history: '',
    default_history: '',
    education: '',
    marital_status: '',
    credit_score: '',
    num_of_loans: '',
    employment_status: '',
    loan_amount: '',
  });
  const [prediction, setPrediction] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const isStepComplete = () => {
    switch (step) {
      case 1:
        return formData.age && formData.income && formData.credit_limit;
      case 2:
        return formData.outstanding_balance && formData.payment_history && formData.default_history;
      case 3:
        return formData.education && formData.marital_status && formData.credit_score;
      case 4:
        return formData.num_of_loans && formData.employment_status && formData.loan_amount;
      default:
        return false;
    }
  };

  const nextStep = () => {
    if (isStepComplete()) {
      setAnimationClass('fade-out');
      setTimeout(() => {
        setStep(step + 1);
        setAnimationClass('fade-in');
      }, 500);
    }
  };

  const handlePrediction = () => {
    const dataToSend = {
      age: parseInt(formData.age),
      income: parseFloat(formData.income),
      credit_limit: parseFloat(formData.credit_limit),
      outstanding_balance: parseFloat(formData.outstanding_balance),
      payment_history: formData.payment_history,
      default_history: formData.default_history,
      education: parseInt(formData.education),
      marital_status: formData.marital_status,
      credit_score: parseInt(formData.credit_score),
      num_of_loans: parseInt(formData.num_of_loans),
      employment_status: formData.employment_status,
      loan_amount: parseFloat(formData.loan_amount)
    };

    // Step 1: Get prediction from Flask API
    axios.post('http://localhost:5001/predict', dataToSend)
      .then((response) => {
        const predictionValue = response.data.prediction;
        setPrediction(predictionValue);  // Store prediction value
        alert('Prediction: ' + (predictionValue === 1 ? 'Default' : 'No Default'));

        // Step 2: Save prediction and form data to MongoDB using Node API
        saveToDatabase(dataToSend, predictionValue);
      })
      .catch((error) => {
        console.error('Error making prediction:', error);
        alert('There was an error making the prediction. Please try again.');
      });
  };

  const saveToDatabase = (data, prediction) => {
    axios.post('http://localhost:5000/save', {
      ...data,
      prediction,
    })
    .then((response) => {
      alert('Data saved successfully: ' + response.data.message);
    })
    .catch((error) => {
      console.error('Error saving data:', error);
    });
  };

  const handleViewAnalysis = () => {
    alert("Viewing analysis of models"); // Placeholder for your logic
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimationClass('');
    }, 500);
    return () => clearTimeout(timer);
  }, [step]);

  return (
    <div className="container">
      <div className="form-container">
        <div className="form-header">Credit Card Default Prediction</div>
        <form id="creditForm">
          {/* Step 1: Personal and Financial Information */}
          <div className={`form-section ${animationClass} ${step >= 1 ? 'visible' : ''}`}>
            <div className="form-section-header">Personal and Financial Information</div>
            <div className="mb-3">
              <label htmlFor="age" className="form-label">Age</label>
              <input
                type="number"
                className="form-control"
                id="age"
                name="age"
                value={formData.age}
                onChange={handleChange}
                required
              />
            </div>
            <div className="mb-3">
              <label htmlFor="income" className="form-label">Income</label>
              <input
                type="number"
                className="form-control"
                id="income"
                name="income"
                value={formData.income}
                onChange={handleChange}
                required
              />
            </div>
            <div className="mb-3">
              <label htmlFor="credit_limit" className="form-label">Credit Limit</label>
              <input
                type="number"
                className="form-control"
                id="credit_limit"
                name="credit_limit"
                value={formData.credit_limit}
                onChange={handleChange}
                required
              />
            </div>
            {step === 1 && (
              <button type="button" className="btn btn-primary mt-3 animated-button" onClick={nextStep}>
                Continue to Account History
              </button>
            )}
          </div>

          {/* Step 2: Account and Payment History */}
          {step >= 2 && (
            <div className={`form-section ${animationClass} ${step >= 2 ? 'visible' : ''}`}>
              <div className="form-section-header">Account and Payment History</div>
              <div className="mb-3">
                <label htmlFor="outstanding_balance" className="form-label">Outstanding Balance</label>
                <input
                  type="number"
                  className="form-control"
                  id="outstanding_balance"
                  name="outstanding_balance"
                  value={formData.outstanding_balance}
                  onChange={handleChange}
                  required
                />
              </div>
              <div className="mb-3">
                <label htmlFor="payment_history" className="form-label">Payment History</label>
                <input
                  type="text"
                  className="form-control"
                  id="payment_history"
                  name="payment_history"
                  value={formData.payment_history}
                  onChange={handleChange}
                  required
                />
              </div>
              <div className="mb-3">
                <label htmlFor="default_history" className="form-label">Default History</label>
                <input
                  type="text"
                  className="form-control"
                  id="default_history"
                  name="default_history"
                  value={formData.default_history}
                  onChange={handleChange}
                  required
                />
              </div>
              {step === 2 && (
                <button type="button" className="btn btn-primary mt-3 animated-button" onClick={nextStep}>
                  Continue to Credit Info
                </button>
              )}
            </div>
          )}

          {/* Step 3: Credit and Personal Information */}
          {step >= 3 && (
            <div className={`form-section ${animationClass} ${step >= 3 ? 'visible' : ''}`}>
              <div className="form-section-header">Credit and Personal Information</div>
              <div className="mb-3">
                <label htmlFor="education" className="form-label">Education Level</label>
                <input
                  type="number"
                  className="form-control"
                  id="education"
                  name="education"
                  value={formData.education}
                  onChange={handleChange}
                  required
                />
              </div>
              <div className="mb-3">
                <label htmlFor="marital_status" className="form-label">Marital Status</label>
                <input
                  type="text"
                  className="form-control"
                  id="marital_status"
                  name="marital_status"
                  value={formData.marital_status}
                  onChange={handleChange}
                  required
                />
              </div>
              <div className="mb-3">
                <label htmlFor="credit_score" className="form-label">Credit Score</label>
                <input
                  type="number"
                  className="form-control"
                  id="credit_score"
                  name="credit_score"
                  value={formData.credit_score}
                  onChange={handleChange}
                  required
                />
              </div>
              {step === 3 && (
                <button type="button" className="btn btn-primary mt-3 animated-button" onClick={nextStep}>
                  Continue to Loan Info
                </button>
              )}
            </div>
          )}

          {/* Step 4: Loan and Employment Information */}
          {step >= 4 && (
            <div className={`form-section ${animationClass} ${step >= 4 ? 'visible' : ''}`}>
              <div className="form-section-header">Loan and Employment Information</div>
              <div className="mb-3">
                <label htmlFor="num_of_loans" className="form-label">Number of Loans</label>
                <input
                  type="number"
                  className="form-control"
                  id="num_of_loans"
                  name="num_of_loans"
                  value={formData.num_of_loans}
                  onChange={handleChange}
                  required
                />
              </div>
              <div className="mb-3">
                <label htmlFor="employment_status" className="form-label">Employment Status</label>
                <input
                  type="text"
                  className="form-control"
                  id="employment_status"
                  name="employment_status"
                  value={formData.employment_status}
                  onChange={handleChange}
                  required
                />
              </div>
              <div className="mb-3">
                <label htmlFor="loan_amount" className="form-label">Loan Amount</label>
                <input
                  type="number"
                  className="form-control"
                  id="loan_amount"
                  name="loan_amount"
                  value={formData.loan_amount}
                  onChange={handleChange}
                  required
                />
              </div>
              {step === 4 && (
                <div className="d-flex justify-content-between">
                  <button type="button" className="btn btn-primary mt-3 animated-button" onClick={handlePrediction}>
                    Submit
                  </button>
                  <button type="button" className="btn btn-secondary mt-3 animated-button" onClick={handleViewAnalysis}>
                    View Analysis of Models
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Display Prediction Result */}
          {prediction !== '' && (
            <div className="prediction-result">
              <h3>Prediction: {prediction === 1 ? 'Default' : 'No Default'}</h3>
            </div>
          )}
        </form>
      </div>
    </div>
  );
}

export default App;
