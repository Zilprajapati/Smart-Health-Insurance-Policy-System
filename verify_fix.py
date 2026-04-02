import pandas as pd
import numpy as np
import pickle
import os
import shap

# Constants
MODEL_DIR = r"d:/Smart_Insurance/models"
CHARGE_MODEL_PATH = os.path.join(MODEL_DIR, "advanced_charge_model.pkl")
DATA_PATH = r"d:/Smart_Insurance/advanced_health_insurance_dataset_10k.csv"

def calculate_health_risk(df):
    score = df['age'].iloc[0] * 0.5
    bmi = df['bmi'].iloc[0]
    if bmi > 30: score += 15
    if bmi > 35: score += 10
    if df['smoker'].iloc[0] == 'yes': score += 30
    medical_cols = ['diabetes', 'hypertension', 'cancer', 'family_heart_disease', 
                    'chronic_kidney_disease', 'asthma', 'thyroid_disorder']
    for col in medical_cols:
        if col in df.columns and df[col].iloc[0] == 1:
            score += 10
    if 'stress_level' in df.columns:
        score += df['stress_level'].iloc[0] * 2
    return min(score, 100)

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
    df['health_risk_score'] = calculate_health_risk(df)
    return df

def verify():
    print("Loading model and data...")
    with open(CHARGE_MODEL_PATH, 'rb') as f:
        charge_model = pickle.load(f)
    df_raw = pd.read_csv(DATA_PATH)
    
    print("Attempting SHAP transformation fix...")
    try:
        # Mimic the logic in app.py
        sample_df = derive_features(df_raw.head(100).copy())
        # The preprocessor is the first step in the pipeline
        prep_data = charge_model.named_steps['preprocessor'].transform(sample_df)
        print(f"Transformation successful! Result shape: {prep_data.shape}")
        
        # Test SHAP explainer
        explainer = shap.TreeExplainer(charge_model.named_steps['regressor'])
        shap_values = explainer.shap_values(prep_data)
        print(f"SHAP values generated successfully! Shape: {shap_values.shape}")
        
        print("\nFix Verified: Missing columns error resolved.")
    except Exception as e:
        print(f"\nVerification Failed: {e}")
        exit(1)

if __name__ == "__main__":
    verify()
