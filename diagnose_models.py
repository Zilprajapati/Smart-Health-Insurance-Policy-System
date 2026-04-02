import pickle
import os
import pandas as pd
import numpy as np

MODEL_DIR = r"d:/Smart_Insurance/models"
CHARGE_MODEL_PATH = os.path.join(MODEL_DIR, "advanced_charge_model.pkl")
RISK_MODEL_PATH = os.path.join(MODEL_DIR, "risk_classifier.pkl")
CLAIM_MODEL_PATH = os.path.join(MODEL_DIR, "claim_probability_model.pkl")

def check_model(path, name):
    print(f"\n{'='*20}")
    print(f"Model: {name}")
    print(f"Path: {path}")
    if not os.path.exists(path):
        print("Status: NOT FOUND")
        return
    
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Type: {type(model)}")
        
        if hasattr(model, 'named_steps'):
            print("Pipeline Steps:", list(model.named_steps.keys()))
            if 'preprocessor' in model.named_steps:
                prep = model.named_steps['preprocessor']
                print("Preprocessor Transformers:")
                for transformer_name, transformer, columns in prep.transformers_:
                    if transformer_name != 'remainder':
                        print(f"  - {transformer_name}: {columns}")
            
            # If it's an XGBoost model, it might have its own feature names
            last_step = list(model.named_steps.values())[-1]
            if hasattr(last_step, 'get_booster'):
                booster = last_step.get_booster()
                print("Internal Feature Names (Top 10):", booster.feature_names[:10])
                print("Total Internal Features:", len(booster.feature_names))
        else:
            print("Model is not a Pipeline.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_model(CHARGE_MODEL_PATH, "Charge Model")
    check_model(RISK_MODEL_PATH, "Risk Model")
    check_model(CLAIM_MODEL_PATH, "Claim Model")
