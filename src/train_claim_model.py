import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Constants
DATA_PATH = r"d:/Smart_Insurance/advanced_health_insurance_dataset_10k.csv"
MODEL_DIR = r"d:/Smart_Insurance/models"
CLAIM_MODEL_PATH = os.path.join(MODEL_DIR, "claim_probability_model.pkl")

def train_claim_model():
    print("Loading data for Claim Model...")
    df = pd.read_csv(DATA_PATH)
    
    # Simulate Claim Target based on Clinical Features
    # Higher age, smoker=yes, higher bmi = higher probability of claim
    np.random.seed(42)
    
    # Simple risk factor calculation
    smoker_risk = (df['smoker'] == 'yes').astype(int) * 0.4
    age_risk = (df['age'] / 100) * 0.3
    bmi_risk = (df['bmi'] / 50) * 0.2
    
    prob = smoker_risk + age_risk + bmi_risk + np.random.rand(len(df)) * 0.2
    df['claim'] = (prob > 0.5).astype(int)
    
    # Features
    categorical_features = [
        'sex', 'smoker', 'region', 'alcohol_consumption', 'physical_activity_level', 
        'diet_type', 'occupation_risk_level', 'income_level', 'residence', 'education_level'
    ]
    numerical_features = [
        'age', 'bmi', 'children', 'heart_rate', 'systolic_bp', 'diastolic_bp', 
        'previous_surgeries', 'sleep_hours', 'stress_level', 'health_risk_score'
    ]
    
    X = df[categorical_features + numerical_features]
    y = df['claim']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, pipeline.predict(X_test))
    print(f"Claim Model Trained. Accuracy: {acc:.4f}")
    
    with open(CLAIM_MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Saved to {CLAIM_MODEL_PATH}")

if __name__ == "__main__":
    train_claim_model()
