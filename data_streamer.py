import pandas as pd
import numpy as np
import requests
import time
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "master_advanced_data.csv")
API_URL = "http://127.0.0.1:8000/predict"

print("System Initialization: Waiting for the API Server to come online (3 seconds)...")
time.sleep(3)
print("PRODUCTION LINE SIMULATOR STARTED. Sensor data stream initialized.")

for chunk in pd.read_csv(DATA_PATH, chunksize=1):
    row = chunk.iloc[0]
    features_raw = row.drop(['Id', 'Response'], errors='ignore').to_dict()
    
    # JSON Type Compatibility Enforcer
    features = {}
    for k, v in features_raw.items():
        if pd.isna(v):
            features[k] = None
        elif isinstance(v, (np.integer, np.floating)):
            features[k] = v.item() 
        else:
            features[k] = v

    part_id = f"PART-{random.randint(10000, 99999)}"
    payload = {"part_id": part_id, "features": features}
    
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            if "DEFECT" in result.get("status", ""):
                print(f"CRITICAL WARNING: {result['part_id']} - Defect Risk: %{result['risk_score']*100:.1f} -> INTERVENTION REQUIRED.")
            else:
                print(f"SUCCESS: {result['part_id']} processed. Status: Normal.")
        else:
            print(f"API SERVER ERROR: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("CONNECTION FAILURE: Target API unreachable. Ensure 'api_server.py' is running.")
        break
        
    time.sleep(2)