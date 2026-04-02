import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from advanced_train import feature_engineering

# Constants
DATA_PATH = r"d:/Smart_Insurance/advanced_health_insurance_dataset_10k.csv"
MODEL_DIR = r"d:/Smart_Insurance/models"
COMPARISON_METRICS_PATH = os.path.join(MODEL_DIR, "comparison_metrics.pkl")

def generate_comparison():
    print("Loading data for Model Comparison...")
    df = pd.read_csv(DATA_PATH)
    
    df = feature_engineering(df)
    
    categorical_features = [
        'sex', 'smoker', 'region', 'alcohol_consumption', 'physical_activity_level', 
        'diet_type', 'occupation_risk_level', 'income_level', 'residence', 'education_level',
        'bmi_category', 'age_group'
    ]
    numerical_features = [
        'age', 'bmi', 'children', 'heart_rate', 'systolic_bp', 'diastolic_bp', 
        'previous_surgeries', 'sleep_hours', 'stress_level',
        'family_heart_disease', 'chronic_kidney_disease', 'asthma', 'thyroid_disorder', 
        'mental_health_condition', 'diabetes', 'hypertension', 'cancer'
    ]
    
    X = df[categorical_features + numerical_features]
    y = df['charges']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    }
    
    comparison_results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        comparison_results[name] = {
            'R2': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
        }
    
    print("Saving comparison metrics...")
    with open(COMPARISON_METRICS_PATH, 'wb') as f:
        pickle.dump(comparison_results, f)
    print(f"Metrics saved to {COMPARISON_METRICS_PATH}")

if __name__ == "__main__":
    generate_comparison()
