import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set constants
DATA_PATH = r"d:/Smart_Insurance/advanced_health_insurance_dataset_10k.csv"
MODEL_PATH = r"d:/Smart_Insurance/models/smartpolicy_model.pkl"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    # Check for duplicates and handle
    print(f"Original shape: {df.shape}")
    df.drop_duplicates(inplace=True)
    print(f"Shape after duplicate removal: {df.shape}")
    return df

def train_model():
    print("Loading data...")
    df = load_data(DATA_PATH)
    
    # Separate features and target
    X = df.drop(columns=['charges'])
    y = df['charges']
    
    # Define features
    categorical_features = [
        'sex', 'smoker', 'region', 'alcohol_consumption', 'physical_activity_level', 
        'diet_type', 'occupation_risk_level', 'income_level', 'residence', 'education_level'
    ]
    
    numerical_features = [
        'age', 'bmi', 'children', 'heart_rate', 'systolic_bp', 'diastolic_bp', 
        'previous_surgeries', 'sleep_hours', 'stress_level', 'health_risk_score',
        'family_heart_disease', 'chronic_kidney_disease', 'asthma', 'thyroid_disorder', 
        'mental_health_condition', 'diabetes', 'hypertension', 'cancer'
        # Note: Binary columns (0/1) like diabetes etc. can be treated as numerical for simplicity
        # or categorical. Treating as numerical is perfectly fine for 0/1.
    ]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Create model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training Random Forest Regressor...")
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Model Performance:")
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    
    # Save model
    print(f"Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully.")

if __name__ == "__main__":
    train_model()
