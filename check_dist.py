import pandas as pd
import numpy as np

DATA_PATH = r"d:/Smart_Insurance/advanced_health_insurance_dataset_10k.csv"

def check():
    df = pd.read_csv(DATA_PATH)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    if 'health_risk_score' not in df.columns:
        print("Error: health_risk_score not in columns")
        return

    print("\nHealth Risk Score Stats:")
    print(df['health_risk_score'].describe())
    
    df['risk_category'] = pd.cut(
        df['health_risk_score'],
        bins=[-np.inf, 30, 60, np.inf],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    print("\nRisk Category Distribution:")
    print(df['risk_category'].value_counts())
    
    # Check if 'High Risk' is dominant
    print("\nSample of data for healthy-looking rows:")
    # healthy: low age, low bmi, no smoker
    healthy_sample = df[(df['age'] < 30) & (df['bmi'] < 25) & (df['smoker'] == 'no')]
    print(f"Healthy sample size: {len(healthy_sample)}")
    if len(healthy_sample) > 0:
        print(healthy_sample[['age', 'bmi', 'smoker', 'health_risk_score']].head())
        # Check their risk categories
        healthy_sample_categories = pd.cut(
            healthy_sample['health_risk_score'],
            bins=[-np.inf, 30, 60, np.inf],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        print("\nHealthy Sample Risk Categories:")
        print(healthy_sample_categories.value_counts())

if __name__ == "__main__":
    check()
