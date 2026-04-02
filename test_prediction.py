import pickle
import pandas as pd
import numpy as np
import os

MODEL_DIR = r"d:/Smart_Insurance/models"
RISK_MODEL_PATH = os.path.join(MODEL_DIR, "risk_classifier.pkl")

def derive_features(df):
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

def test():
    with open(RISK_MODEL_PATH, 'rb') as f:
        risk_model = pickle.load(f)

    # Healthy profile
    healthy_dict = {
        'age': [25], 'sex': ['male'], 'bmi': [22.0], 'children': [0], 'smoker': ['no'],
        'region': ['south'], 'alcohol_consumption': ['never'], 
        'physical_activity_level': ['high'], 'diet_type': ['healthy'],
        'occupation_risk_level': ['low'], 'income_level': ['high'],
        'residence': ['urban'], 'education_level': ['master'],
        'heart_rate': [70], 'systolic_bp': [120], 'diastolic_bp': [80],
        'previous_surgeries': [0], 'sleep_hours': [8.0], 
        'stress_level': [1], 'health_risk_score': [10.0], # Note: this was dropped in training
        'diabetes': [0], 'hypertension': [0], 'cancer': [0],
        'family_heart_disease': [0], 'chronic_kidney_disease': [0],
        'asthma': [0], 'thyroid_disorder': [0], 'mental_health_condition': [0]
    }
    healthy_df = derive_features(pd.DataFrame(healthy_dict))
    
    # Unhealthy profile
    unhealthy_dict = healthy_dict.copy()
    unhealthy_dict.update({
        'smoker': ['yes'], 'bmi': [35.0], 'diabetes': [1], 'hypertension': [1], 
        'stress_level': [9], 'health_risk_score': [90.0]
    })
    unhealthy_df = derive_features(pd.DataFrame(unhealthy_dict))

    print(f"Healthy Pred: {risk_model.predict(healthy_df)[0]}")
    print(f"Unhealthy Pred: {risk_model.predict(unhealthy_df)[0]}")

if __name__ == "__main__":
    test()
