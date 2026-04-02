import pandas as pd
import numpy as np
import pickle
import os

# Paths
MODEL_DIR = r"d:/Smart_Insurance/models"
RISK_MODEL_PATH = os.path.join(MODEL_DIR, "risk_classifier.pkl")
DATA_PATH = r"d:/Smart_Insurance/advanced_health_insurance_dataset_10k.csv"

# Load Model
with open(RISK_MODEL_PATH, 'rb') as f:
    risk_model = pickle.load(f)

# Load Data to get categories for valid inputs
df_raw = pd.read_csv(DATA_PATH)

def derive_features(df):
    # Same logic as app.py
    df['bmi_category'] = pd.cut(
        df['bmi'], 
        bins=[-np.inf, 18.5, 24.9, 29.9, np.inf], 
        labels=['Underweight', 'Normal', 'Overweight', 'Obese']
    )
    df['age_group'] = pd.cut(
        df['age'], 
        bins=[-np.inf, 35, 55, np.inf], 
        labels=['Young', 'Adult', 'Senior']
    )
    return df

# Create Test Cases
# Case 1: Healthy (Young, Normal BMI, Non-smoker, No diseases)
case1 = {
    'age': [25], 'sex': ['male'], 'bmi': [22.0], 'children': [0], 'smoker': ['no'],
    'region': [df_raw['region'].unique()[0]], 
    'alcohol_consumption': ['None'], 
    'physical_activity_level': ['Active'], 
    'diet_type': ['Vegetarian'],
    'occupation_risk_level': ['Low'], 
    'income_level': ['Medium'],
    'residence': ['Urban'], 
    'education_level': ['Bachelor'],
    'heart_rate': [70], 'systolic_bp': [115], 'diastolic_bp': [75],
    'previous_surgeries': [0], 'sleep_hours': [8], 
    'stress_level': [3], 'health_risk_score': [20], # Should be Low Risk if used, but ignored
    'diabetes': [0], 'hypertension': [0], 'cancer': [0],
    'family_heart_disease': [0], 'chronic_kidney_disease': [0],
    'asthma': [0], 'thyroid_disorder': [0], 'mental_health_condition': [0]
}

# Case 2: Unhealthy (Senior, Obese, Smoker, Diseases)
case2 = {
    'age': [65], 'sex': ['male'], 'bmi': [35.0], 'children': [0], 'smoker': ['yes'],
    'region': [df_raw['region'].unique()[0]], 
    'alcohol_consumption': ['High'], 
    'physical_activity_level': ['Sedentary'], 
    'diet_type': ['Paleo'], # whatever
    'occupation_risk_level': ['High'], 
    'income_level': ['Medium'],
    'residence': ['Urban'], 
    'education_level': ['Bachelor'],
    'heart_rate': [90], 'systolic_bp': [150], 'diastolic_bp': [95],
    'previous_surgeries': [2], 'sleep_hours': [5], 
    'stress_level': [9], 'health_risk_score': [85], # Should be High Risk
    'diabetes': [1], 'hypertension': [1], 'cancer': [0],
    'family_heart_disease': [1], 'chronic_kidney_disease': [0],
    'asthma': [0], 'thyroid_disorder': [0], 'mental_health_condition': [0]
}

df1 = pd.DataFrame(case1)
df1 = derive_features(df1)

df2 = pd.DataFrame(case2)
df2 = derive_features(df2)

print("Predicting Case 1 (Healthy)...")
try:
    pred1 = risk_model.predict(df1)[0]
    print(f"Prediction 1: {pred1}")
except Exception as e:
    print(f"Error 1: {e}")

print("\nPredicting Case 2 (Unhealthy)...")
try:
    pred2 = risk_model.predict(df2)[0]
    print(f"Prediction 2: {pred2}")
except Exception as e:
    print(f"Error 2: {e}")
