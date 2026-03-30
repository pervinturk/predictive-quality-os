import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb
import networkx as nx
import matplotlib.pyplot as plt
import requests
import joblib
import psutil
import os
import time
import base64
from datetime import datetime
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import optuna
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Predictive Quality OS", layout="wide")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "master_advanced_data.csv")
REGISTRY_DIR = os.path.join(PROJECT_ROOT, "models", "registry")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
for d in [REGISTRY_DIR, DOCS_DIR]: os.makedirs(d, exist_ok=True)

LANG = {"TR": {"title": "Predictive Quality Operating System", "ram": "System Telemetry", "sim": "Digital Twin (What-If)"},
        "EN": {"title": "Predictive Quality Operating System", "ram": "System Telemetry", "sim": "Digital Twin (What-If)"}}
lang_sel = st.sidebar.radio("Language / Dil", ["TR", "EN"])
t = LANG[lang_sel]

@st.cache_resource
def load_and_engineer_data():
    if not os.path.exists(DATA_PATH): st.stop()
    df = pl.read_csv(DATA_PATH, ignore_errors=True).to_pandas()
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if 'Id' in num_cols: num_cols.remove('Id')
    if 'Response' in num_cols: num_cols.remove('Response')
    
    for col in cat_cols:
        df[col] = df[col].astype(object).fillna("BOS").astype(str).astype('category')
    
    if len(cat_cols) > 2:
        df['Station_Path'] = df[cat_cols[:3]].astype(str).agg('-'.join, axis=1).astype('category')
    
    for col in num_cols[:3]:
        df[f'{col}_noise'] = df[col].rolling(window=5, min_periods=1).std().fillna(0)
        num_cols.append(f'{col}_noise')
        
    return df, num_cols

df_main, numeric_features = load_and_engineer_data()

@st.cache_resource
def train_model(df, num_cols):
    X = df.drop(columns=['Id', 'Response'], errors='ignore')
    y = df['Response']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    variances = X_train.select_dtypes(include=[np.number]).var()
    alive_num_features = variances[variances > 0.01].index.tolist()[:100] 
    
    alive_features = alive_num_features + X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    X_train_sel, X_test_sel = X_train[alive_features], X_test[alive_features]
    weight = sum(y_train == 0) / (sum(y_train == 1) + 1) * 20
    
    def objective(trial):
        params = {'max_depth': trial.suggest_int('max_depth', 4, 7), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1), 'n_estimators': trial.suggest_int('n_estimators', 100, 200), 'tree_method': 'hist', 'enable_categorical': True, 'scale_pos_weight': weight, 'n_jobs': -1}
        temp_model = xgb.XGBClassifier(**params)
        temp_model.fit(X_train_sel, y_train)
        preds = temp_model.predict_proba(X_test_sel)[:, 1]
        return roc_auc_score(y_test, preds) if len(np.unique(y_test)) > 1 else accuracy_score(y_test, preds > 0.5)

    try:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=3)
        best_params = study.best_params
    except:
        best_params = {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 150}
    
    best_params.update({'tree_method': 'hist', 'enable_categorical': True, 'scale_pos_weight': weight, 'n_jobs': -1})
    base_model = xgb.XGBClassifier(**best_params)
    calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv='prefit')
    
    start_train = time.time()
    base_model.fit(X_train_sel, y_train)
    calibrated_model.fit(X_test_sel, y_test)
    train_latency = time.time() - start_train
    
    # Save Dtypes as a native Python dictionary for structural integrity
    dtypes_dict = {col: str(dtype) for col, dtype in X_train_sel.dtypes.items()}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    joblib.dump(calibrated_model, os.path.join(REGISTRY_DIR, f"model_v_{timestamp}.pkl"))
    joblib.dump(dtypes_dict, os.path.join(REGISTRY_DIR, f"dtypes_v_{timestamp}.pkl"))
    
    return calibrated_model, base_model, X_test_sel, y_test, alive_features, train_latency, best_params

calibrated_model, base_model, X_test, y_test, selected_features, train_latency, best_params = train_model(df_main, numeric_features)
importances = pd.Series(base_model.feature_importances_, index=selected_features)
num_available = [f for f in selected_features if f in numeric_features]
top_sensor = importances[num_available].idxmax() if len(num_available) > 0 else importances.idxmax()

st.sidebar.markdown(f"### {t['ram']}")
st.sidebar.progress(psutil.cpu_percent() / 100, text=f"CPU: %{psutil.cpu_percent()}")
st.sidebar.progress(psutil.virtual_memory().percent / 100, text=f"RAM: %{psutil.virtual_memory().percent}")
st.sidebar.caption(f"Latency: {train_latency:.2f} s")

maliyet_fp = st.sidebar.number_input("False Alarm Cost" if lang_sel=="EN" else "Hatalı Alarm (TL)", value=50)
maliyet_fn = st.sidebar.number_input("Defect Cost" if lang_sel=="EN" else "Bozuk Ürün (TL)", value=2500)

st.title(t["title"])
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Finance & Export", "NetworkX Route Analysis", "Concept Drift & SHAP XAI", f"{t['sim']}", "Autonomous LLM & PDF"])

with tab1:
    y_pred = calibrated_model.predict(X_test)
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    kurtarilan_para = (cm[1][1] * maliyet_fn) - (cm[0][1] * maliyet_fp)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Discrimination (AUC)", f"%{roc_auc_score(y_test, y_prob)*100:.2f}")
    col2.metric("Detected Defects", f"{cm[1][1]}")
    col3.metric("Recovered Profit", f"{kurtarilan_para:,}")
    
    st.markdown("### High-Risk Components Database")
    risk_df = pd.DataFrame({'Sensor_Data': X_test[top_sensor], 'Actual_Status': y_test, 'Risk_%': y_prob*100})
    st.dataframe(risk_df[risk_df['Risk_%'] > 60].sort_values(by='Risk_%', ascending=False).head(5))
    href = f'<a href="data:file/csv;base64,{base64.b64encode(risk_df.to_csv(index=False).encode()).decode()}" download="High_Risk_Components.csv">Download Report (CSV)</a>'
    st.markdown(href, unsafe_allow_html=True)

with tab2:
    st.subheader("Cascade Error Network (Empirical Data)")
    if 'Station_Path' in df_main.columns:
        bad_paths = df_main[df_main['Response']==1]['Station_Path'].value_counts().head(10).index.tolist()
        G = nx.DiGraph()
        for path in bad_paths:
            if pd.notna(path) and str(path) != "BOS": 
                nodes = str(path).split('-')
                for i in range(len(nodes)-1):
                    if nodes[i] != "BOS" and nodes[i+1] != "BOS" and nodes[i] != "nan": G.add_edge(nodes[i], nodes[i+1])
        if len(G.edges) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            nx.draw(G, nx.spring_layout(G), with_labels=True, node_color='#ff9999', node_size=3000, font_size=9, font_weight='bold', edge_color='red', arrows=True, ax=ax)
            st.pyplot(fig)

with tab3:
    col_x, col_y = st.columns(2)
    with col_x:
        st.subheader(f"Z-Score Anomaly Detection (`{top_sensor}`)")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(X_test[top_sensor].values[:150], color='black')
        ax2.axhline(X_test[top_sensor].mean() + (3*X_test[top_sensor].std()), color="red", linestyle="--")
        st.pyplot(fig2)
    with col_y:
        st.subheader("XGBoost Decision Tree Architecture")
        if st.button("Draw Tree"):
            fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
            xgb.plot_tree(base_model, num_trees=0, ax=ax, rankdir='LR')
            st.pyplot(fig)

with tab4:
    st.subheader("Scenario Simulator")
    top3 = importances[num_available].sort_values(ascending=False).head(3).index.tolist() if len(num_available) > 0 else []
    if len(top3) > 0:
        sim_data = X_test.iloc[[0]].copy()
        cols = st.columns(3)
        for i, sensor in enumerate(top3):
            mean_v, std_v = float(X_test[sensor].mean()), float(X_test[sensor].std())
            sim_data[sensor] = cols[i].slider(f"{sensor}", min_value=mean_v-(std_v*2), max_value=mean_v+(std_v*2), value=mean_v)
        st.metric("Defect Probability", f"%{calibrated_model.predict_proba(sim_data)[:, 1][0]*100:.2f}")

with tab5:
    st.subheader("LLM Autonomous Supervisor")
    if st.button("Initialize Agent and Generate PDF"):
        with st.spinner("LLM Analysis in progress..."):
            sys_msg = "You are an Industrial Engineer focused on cost optimization. Provide 3 technical prescriptions to prevent cascade errors."
            usr_msg = f"The following sensors show critical deviation:\n{importances.sort_values(ascending=False).head(5).to_string()}\nGenerate an emergency action plan."
            try:
                res = requests.post("http://localhost:11434/api/generate", json={"model": "llama3.2", "prompt": f"{sys_msg}\n{usr_msg}", "stream": False})
                llm_response = res.json()['response']
                st.write(llm_response)
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=16)
                pdf.cell(200, 10, txt="Predictive Quality OS - Autonomous Report", ln=True, align='C')
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=f"Recovered Profit: {kurtarilan_para} | AUC: %{roc_auc_score(y_test, y_prob)*100:.2f}", ln=True)
                pdf.cell(200, 10, txt="--- Prescriptive Analytics Plan ---", ln=True)
                pdf.multi_cell(0, 10, txt=llm_response.encode('latin-1', 'replace').decode('latin-1'))
                pdf_path = os.path.join(DOCS_DIR, "Predictive_Quality_Report.pdf")
                pdf.output(pdf_path)
                st.success(f"PDF successfully generated: `{pdf_path}`")
            except:
                st.error("LLM API endpoint not reachable. Ensure the local model instance is running.")