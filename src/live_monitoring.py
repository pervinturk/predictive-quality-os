import streamlit as st
import pandas as pd
import requests
import time

st.set_page_config(page_title="Live Monitoring Room", layout="wide")
st.title("Predictive Quality Control Room")

API_LOGS_URL = "http://127.0.0.1:8000/logs?limit=15"
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Production Line Telemetry")
    table_placeholder = st.empty()
    
with col2:
    st.subheader("Alert Panel")
    alert_placeholder = st.empty()

try:
    res = requests.get(API_LOGS_URL) 
    if res.status_code == 200:
        logs = res.json()
        df_logs = pd.DataFrame(logs)
        
        if not df_logs.empty:
            df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp']).dt.strftime('%H:%M:%S')
            df_logs['risk_score'] = (df_logs['risk_score'] * 100).round(2).astype(str) + "%"
            table_placeholder.dataframe(df_logs[['timestamp', 'part_id', 'status', 'risk_score']])
            
            latest = df_logs.iloc[0]
            if "DEFECT" in latest['status']:
                alert_placeholder.error(f"ACTION REQUIRED!\nComponent: {latest['part_id']}\nRisk Level: {latest['risk_score']}")
            else:
                alert_placeholder.success(f"PRODUCTION NORMAL\nComponent: {latest['part_id']}\nRisk Level: {latest['risk_score']}")
        else:
            table_placeholder.info("Production line is idle. Execute 'data_streamer.py' to begin simulation.")
    else:
        table_placeholder.error(f"Database API Error: {res.status_code}")
except Exception:
    table_placeholder.error("API Connection Failed. Please ensure 'api_server.py' via uvicorn is running.")

time.sleep(2)
st.rerun()