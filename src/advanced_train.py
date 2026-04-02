import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
from xgboost import XGBRegressor

# Constants
DATA_PATH = r"d:/Smart_Insurance/advanced_health_insurance_dataset_10k.csv"
MODEL_DIR = r"d:/Smart_Insurance/models"
CHARGE_MODEL_PATH = os.path.join(MODEL_DIR, "advanced_charge_model.pkl")
RISK_MODEL_PATH = os.path.join(MODEL_DIR, "risk_classifier.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    df.drop_duplicates(inplace=True)
    return df

def feature_engineering(df):
    print("Performing Feature Engineering...")
    
    # Check if health_risk_score is constant or missing
    if 'health_risk_score' not in df.columns or df['health_risk_score'].nunique() <= 1:
        print("Regenerating dynamic health_risk_score for training...")
        # Simple weighted score calculation for training diversity
        # Base score starts from age / 2
        df['health_risk_score'] = df['age'] * 0.5
        
        # BMI impact
        df.loc[df['bmi'] > 30, 'health_risk_score'] += 15
        df.loc[df['bmi'] > 35, 'health_risk_score'] += 10
        
        # Smoking impact
        df.loc[df['smoker'] == 'yes', 'health_risk_score'] += 30
        
        # Medical conditions impact
        medical_cols = ['diabetes', 'hypertension', 'cancer', 'family_heart_disease', 
                        'chronic_kidney_disease', 'asthma', 'thyroid_disorder']
        for col in medical_cols:
            if col in df.columns:
                # Assuming 1 is Yes, 0 is No
                df.loc[df[col] == 1, 'health_risk_score'] += 10
        
        # Lifestyle impact
        if 'stress_level' in df.columns:
            df['health_risk_score'] += df['stress_level'] * 2
        
        # Clip to 0-100
        df['health_risk_score'] = df['health_risk_score'].clip(0, 100)

    # BMI Category
    df['bmi_category'] = pd.cut(
        df['bmi'], 
        bins=[-np.inf, 18.5, 24.9, 29.9, np.inf], 
        labels=['Underweight', 'Normal', 'Overweight', 'Obese']
    )
    
    # Age Group
    df['age_group'] = pd.cut(
        df['age'], 
        bins=[-np.inf, 35, 55, np.inf], 
        labels=['Young', 'Adult', 'Senior']
    )
    
    # Risk Category (Target for Classification)
    # Low (0-40), Medium (41-70), High (71-100)
    df['risk_category'] = pd.cut(
        df['health_risk_score'],
        bins=[-np.inf, 40, 70, np.inf],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    return df

def train_models():
    print("Loading Data...")
    df = load_data(DATA_PATH)
    
    df = feature_engineering(df)
    
    # --- PREPARE DATA ---
    X = df.drop(columns=['charges', 'risk_category', 'health_risk_score']) # Drop targets and direct proxy
    y_charge = df['charges']
    y_risk = df['risk_category']
    
    # Define features types
    categorical_features = [
        'sex', 'smoker', 'region', 'alcohol_consumption', 'physical_activity_level', 
        'diet_type', 'occupation_risk_level', 'income_level', 'residence', 'education_level',
        'bmi_category', 'age_group',
        # Binary cols can be treated as cat or num. Treating 'No'/'Yes' as cat if they were strings, 
        # but in this dataset binary/medical cols might be int 0/1 or strings. 
        # Checking dataset content from previous turns: 
        # 'diabetes', 'hypertension' ... are 0/1 integers.
        # 'smoker' is 'yes'/'no'.
    ]
    
    # Update categorical list based on actual dataframe types if needed, but we known the schema.
    # The new engineered features are categorical.
    
    numerical_features = [
        'age', 'bmi', 'children', 'heart_rate', 'systolic_bp', 'diastolic_bp', 
        'previous_surgeries', 'sleep_hours', 'stress_level',
        'family_heart_disease', 'chronic_kidney_disease', 'asthma', 'thyroid_disorder', 
        'mental_health_condition', 'diabetes', 'hypertension', 'cancer'
    ]

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # --- CHARGE PREDICTION MODEL (XGBoost) ---
    print("\nTraining Charge Prediction Model (XGBoost)...")
    charge_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_charge, test_size=0.2, random_state=42)
    charge_pipeline.fit(X_train, y_train)
    
    y_pred = charge_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Charge Model -> R2: {r2:.4f}, RMSE: {rmse:.2f}")
    
    # --- RISK CLASSIFICATION MODEL (Random Forest) ---
    print("\nTraining Risk Classification Model (Random Forest)...")
    risk_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_risk, test_size=0.2, random_state=42, stratify=y_risk)
    risk_pipeline.fit(X_train_r, y_train_r)
    
    y_pred_r = risk_pipeline.predict(X_test_r)
    acc = accuracy_score(y_test_r, y_pred_r)
    print(f"Risk Model -> Accuracy: {acc:.4f}")
    print(classification_report(y_test_r, y_pred_r))
    
    # --- SAVE ARTIFACTS ---
    print("Saving models...")
    with open(CHARGE_MODEL_PATH, 'wb') as f:
        pickle.dump(charge_pipeline, f)
        
    with open(RISK_MODEL_PATH, 'wb') as f:
        pickle.dump(risk_pipeline, f)
        
    # Save metadata for app
    metadata = {
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'model_performance': {'r2': r2, 'rmse': rmse, 'accuracy': acc}
    }
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
        
    print("All models saved successfully.")

if __name__ == "__main__":
    train_models()
