# Predictive Quality Operating System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Microservice-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-AI_Engine-yellow.svg)
![Optuna](https://img.shields.io/badge/Optuna-AutoML-blueviolet.svg)
![Llama3](https://img.shields.io/badge/Llama_3.2-Prescriptive_AI-orange.svg)

## Executive Summary
**[Active Development]** An autonomous Industry 4.0 Decision Support System featuring real-time anomaly detection, cost-sensitive AutoML (XGBoost), and LLM-driven prescriptive analytics.

<img width="1907" height="889" alt="a (1)" src="https://github.com/user-attachments/assets/4c7d84c8-f091-4da7-8461-2a806bd4093c" />
<img width="1907" height="890" alt="a (2)" src="https://github.com/user-attachments/assets/2d1df417-5a5d-46fa-98b4-994991601aa8" />
<img width="1907" height="891" alt="a (3)" src="https://github.com/user-attachments/assets/b79957df-4caa-4f8b-84f4-42e6147501f3" />
<img width="1907" height="891" alt="a (4)" src="https://github.com/user-attachments/assets/3c64bef9-2e39-4af1-afc6-ceaf9f893a1a" />
<img width="1907" height="892" alt="a (5)" src="https://github.com/user-attachments/assets/106a2973-8465-4dc9-a0e3-1a63552f67cb" />
<img width="1907" height="891" alt="a (6)" src="https://github.com/user-attachments/assets/b24fb254-a934-445b-88ec-979a199cd99d" />

The Predictive Quality Operating System is a microservices-based architecture designed for modern manufacturing environments. It shifts quality control from a reactive state to a predictive intervention model by ingesting high-frequency sensor data and detecting cascading errors within milliseconds. The system prioritizes minimizing the financial impact of false negatives (defective components reaching assembly) over false positives.

## System Architecture & Microservices
The ecosystem simulates a decoupled smart factory environment utilizing four distinct microservices:

### 1. Model Training & Analytics Center (main_dashboard.py)
* **Data Ingestion:** Utilizes RAM-optimized data reading via Polars to handle large-scale industrial datasets.
* **AutoML Engine:** Employs Optuna for autonomous hyperparameter optimization to construct accurate XGBoost classifier models.
* **Cost-Sensitive Learning:** The algorithm mathematically weights instances to penalize false negatives, optimizing directly for recovered profit.
* **Autonomous LLM Supervisor:** Integrates a local Llama 3.2 instance to translate statistical feature importances into prescriptive action plans, exporting them as PDF reports via FPDF.

### 2. Central AI Prediction Engine (api_server.py)
* **High-Speed Inference:** A FastAPI-driven endpoint that loads serialized models and strict dictionary schemas into memory for low-latency predictions.
* **Strict Schema Validation:** Enforces structural integrity by reconstructing dynamic station routes ("Station_Path") and mapping incoming JSON payloads to the exact training blueprint.
* **Thread-Safe Logging:** Utilizes SQLAlchemy with a thread-safe SQLite backend to log predictions, timestamps, and component IDs asynchronously.

### 3. Real-Time Telemetry Dashboard (live_monitoring.py)
* **Role-Based UX:** Features distinct viewing modes for "Operators" (focused on high-visibility tables and pulsing alerts) and "Engineers" (focused on continuous trend analysis and advanced logs).
* **Dynamic Threshold Control:** Allows engineers to adjust the critical risk threshold in real-time via session state management, dynamically altering conditional formatting rules without restarting the server.
* **Control Room Interface:** A highly optimized, auto-refreshing Streamlit application that continuously polls the API for production logs with zero interface locking.

### 4. Production Line Simulator (data_streamer.py)
* **Data Streaming:** Acts as a continuous data pump, reading the master dataset iteratively and translating Numpy data types into pure JSON formats for HTTP POST requests.
* **Load Testing:** Simulates realistic factory constraints by injecting asynchronous payloads into the FastAPI endpoint.

## Installation & Setup

Ensure Python 3.9+ is installed, alongside a local instance of Ollama (for LLM reporting features) and the Graphviz engine (for tree visualization).

```bash
# Clone the repository
git clone https://github.com/pervinturk/predictive-quality-os.git
cd predictive-quality-os

# Install dependencies
pip install -r requirements.txt
```
## Execution Guide
**Prerequisite (Local LLM Engine):**
Ensure the Ollama service is running in the background with the required model before starting the simulation.
```bash
ollama run llama3.2
```

To accurately simulate the microservices environment, initialize the system in the exact sequence below using separate terminal instances.
Train Model & Generate Schema (Terminal 1):
```bash
streamlit run src/main_dashboard.py
```
Initialize Prediction API (Terminal 2):
```bash
uvicorn src.api_server:app --reload
```
Launch Live Monitoring Room (Terminal 3):
```bash
streamlit run src/live_monitoring.py
```
Start Production Simulator (Terminal 4):
```bash
python src/data_streamer.py
```

## Development Roadmap (Active)
This project follows an agile methodology. Current progress:

- [x] **Sprint 1 (Completed):** Core AutoML engine, strict type validation API, Real-time Streamlit telemetry, and Role-based UX/UI.
- [ ] **Sprint 2 (In Progress):** OEE (Overall Equipment Effectiveness) integration, NetworkX Bottleneck Analysis optimization, and Root Cause visualization.
- [ ] **Sprint 3 (Planned):** Docker containerization, JWT-based API security, and shift-end automated reporting.

---
*Disclaimer: This project is an independent professional portfolio work designed to demonstrate advanced Industry 4.0 software architecture, cost-sensitive machine learning, and decoupled microservices. It is not affiliated with, endorsed by, or sponsored by any specific corporation. Any resemblance to proprietary enterprise software is purely coincidental.*
