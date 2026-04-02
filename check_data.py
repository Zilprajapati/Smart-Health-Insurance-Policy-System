import pandas as pd
import numpy as np

DATA_PATH = r"d:/Smart_Insurance/advanced_health_insurance_dataset_10k.csv"

try:
    df = pd.read_csv(DATA_PATH)
    print("Dataset Loaded.")
    print(f"Shape: {df.shape}")
    
    # Re-create the target exactly as in training
    df['risk_category'] = pd.cut(
        df['health_risk_score'],
        bins=[-np.inf, 30, 60, np.inf],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    print("\nRisk Category Distribution:")
    print(df['risk_category'].value_counts(normalize=True))
    
    print("\nRisk Category Counts:")
    print(df['risk_category'].value_counts())
    
except Exception as e:
    print(f"Error: {e}")
