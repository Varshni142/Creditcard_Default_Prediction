import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pickle
import joblib

# Load your dataset (Replace with your actual CSV file path)
csv_file_path = 'CCP.csv'  # Path to your CSV file
df = pd.read_csv(csv_file_path)

# Check the columns in the DataFrame
print("Columns in the DataFrame:", df.columns.tolist())

# Map categorical variables to numerical values
# Mapping 'SEX'
if 'SEX' in df.columns:
    df['SEX'] = df['SEX'].map({'M': 1, 'F': 0})  # Male = 1, Female = 0

# Mapping 'MARRIAGE'
if 'MARRIAGE' in df.columns:
    df['MARRIAGE'] = df['MARRIAGE'].map({'Married': 1, 'Single': 0})

# Mapping 'EDUCATION'
if 'EDUCATION' in df.columns:
    df['EDUCATION'] = df['EDUCATION'].map({
        'University': 2,  # Assign values as needed
        'Graduate School': 3,
        'High School': 1,
        'Others': 0
    })

# Mapping 'default ' (with trailing space)
if 'default ' in df.columns:
    df['default '] = df['default '].map({'Y': 1, 'N': 0})

# Fill missing values with 0
df.fillna(0, inplace=True)

# Define features (X) and target variable (y)
if 'default ' in df.columns:
    X = df.drop('default ', axis=1)  # Features
    y = df['default ']  # Target variable
else:
    raise ValueError("Column 'default ' not found in DataFrame.")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize scaler and scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and save the Neural Network model
neural_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
neural_model.fit(X_train_scaled, y_train)
with open('neural_credit.pkl', 'wb') as f:
    pickle.dump(neural_model, f)

# Train and save the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
joblib.dump(rf_model, 'rf_credit.pkl')

# Train and save the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)
joblib.dump(dt_model, 'dt_credit.pkl')

# Save the scaler
with open('scaler_credit.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Evaluate the models
neural_pred = neural_model.predict(X_test_scaled)
rf_pred = rf_model.predict(X_test_scaled)
dt_pred = dt_model.predict(X_test_scaled)

# Print classification reports
print("Neural Network Classification Report:\n", classification_report(y_test, neural_pred))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))
print("Decision Tree Classification Report:\n", classification_report(y_test, dt_pred))

print("Models and scaler saved successfully!")
