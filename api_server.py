from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "registry")

if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

# Validation 1: Ensure synchronization between Model and Dtype Dictionary
model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("model_v_")]
valid_timestamps = []
for mf in model_files:
    ts = mf.replace("model_v_", "").replace(".pkl", "")
    if os.path.exists(os.path.join(MODEL_DIR, f"dtypes_v_{ts}.pkl")):
        valid_timestamps.append(ts)

if not valid_timestamps:
    print("CRITICAL ERROR: Compatible Model and Dtype Dictionary not found.")
    print("ACTION REQUIRED: Please execute 'main_dashboard.py' to train a new model instance.")
    sys.exit(1)

latest_ts = sorted(valid_timestamps)[-1]
MODEL_PATH = os.path.join(MODEL_DIR, f"model_v_{latest_ts}.pkl")
DTYPES_PATH = os.path.join(MODEL_DIR, f"dtypes_v_{latest_ts}.pkl")

# Configuration: Thread-Safe SQLite Initialization
DB_PATH = os.path.join(PROJECT_ROOT, "data", "factory_logs.db")
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ProductionLog(Base):
    __tablename__ = "production_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    part_id = Column(String, index=True)
    risk_score = Column(Float)
    status = Column(String)

Base.metadata.create_all(bind=engine)
app = FastAPI(title="Predictive Quality AI Engine API", version="1.0")

try:
    ai_model = joblib.load(MODEL_PATH)
    saved_dtypes = joblib.load(DTYPES_PATH)
    print(f"System initialized successfully. Active Model Version: {latest_ts}")
except Exception as e:
    print(f"System Initialization Failure: {e}")
    sys.exit(1)

class SensorData(BaseModel):
    part_id: str
    features: dict

@app.post("/predict")
def predict_defect(data: SensorData):
    try:
        df_live = pd.DataFrame([data.features])
        
        # Validation 2: Cascade Route Generation based on Training Blueprint
        cat_cols = [col for col, dtype in saved_dtypes.items() if dtype == 'category' and col != 'Station_Path']
        if len(cat_cols) >= 3:
            path_cols = cat_cols[:3]
            for c in path_cols:
                if c not in df_live.columns: df_live[c] = "BOS"
            df_live['Station_Path'] = df_live[path_cols].fillna("BOS").astype(str).agg('-'.join, axis=1)

        # Validation 3: Strict Column Synchronization
        expected_features = list(saved_dtypes.keys())
        for col in expected_features:
            if col not in df_live.columns:
                df_live[col] = np.nan
        df_live = df_live[expected_features] 
        
        # Validation 4: Type Enforcement
        for col, dtype_str in saved_dtypes.items():
            if dtype_str == 'category':
                df_live[col] = df_live[col].fillna("BOS").astype(str).astype('category')
            else:
                df_live[col] = pd.to_numeric(df_live[col], errors='coerce').fillna(0.0).astype(dtype_str)

        risk_prob = ai_model.predict_proba(df_live)[:, 1][0]
        status = "DEFECT RISK" if risk_prob > 0.60 else "NORMAL"
        
        db = SessionLocal()
        try:
            new_log = ProductionLog(part_id=data.part_id, risk_score=float(risk_prob), status=status)
            db.add(new_log)
            db.commit()
        finally:
            db.close()
        
        return {"part_id": data.part_id, "risk_score": float(risk_prob), "status": status}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs")
def get_logs(limit: int = 15):
    db = SessionLocal()
    try:
        logs = db.query(ProductionLog).order_by(ProductionLog.timestamp.desc()).limit(limit).all()
        return [{"timestamp": log.timestamp.isoformat(), "part_id": log.part_id, "risk_score": log.risk_score, "status": log.status} for log in logs]
    finally:
        db.close()