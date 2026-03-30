# Predictive Quality Operating System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Microservice-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-AI_Engine-yellow.svg)
![Optuna](https://img.shields.io/badge/Optuna-AutoML-blueviolet.svg)
![Llama3](https://img.shields.io/badge/Llama_3.2-Prescriptive_AI-orange.svg)

## Executive Summary
**[Active Development]** An autonomous Industry 4.0 Decision Support System featuring real-time anomaly detection, cost-sensitive AutoML (XGBoost), and LLM-driven prescriptive analytics.

The Predictive Quality Operating System is a microservices-based architecture designed for modern manufacturing environments. It shifts quality control from a reactive state to a predictive intervention model by ingesting high-frequency sensor data and detecting cascading errors within milliseconds. The system prioritizes minimizing the financial impact of false negatives (defective components reaching assembly) over false positives.

## System Architecture & Microservices
The ecosystem simulates a decoupled smart factory environment utilizing four distinct microservices:

### 1. Model Training & Analytics Center (`main_dashboard.py`)
* **Data Ingestion:** Utilizes RAM-optimized data reading via Polars to handle large-scale industrial datasets.
* **AutoML Engine:** Employs Optuna for autonomous hyperparameter optimization to construct accurate XGBoost classifier models.
* **Cost-Sensitive Learning:** The algorithm mathematically weights instances to penalize false negatives, optimizing directly for recovered profit.
* **Autonomous LLM Supervisor:** Integrates a local Llama 3.2 instance to translate statistical feature importances into prescriptive action plans, exporting them as PDF reports via FPDF.

### 2. Central AI Prediction Engine (`api_server.py`)
* **High-Speed Inference:** A FastAPI-driven endpoint that loads serialized models and strict dictionary schemas into memory for low-latency predictions.
* **Strict Schema Validation:** Enforces structural integrity by reconstructing dynamic station routes ("Station_Path") and mapping incoming JSON payloads to the exact training blueprint.
* **Thread-Safe Logging:** Utilizes SQLAlchemy with a thread-safe SQLite backend to log predictions, timestamps, and component IDs asynchronously.

### 3. Real-Time Telemetry Dashboard (`live_monitoring.py`)
* **Role-Based UX:** Features distinct viewing modes for "Operators" (focused on high-visibility tables and pulsing alerts) and "Engineers" (focused on continuous trend analysis and advanced logs).
* **Dynamic Threshold Control:** Allows engineers to adjust the critical risk threshold in real-time via session state management, dynamically altering conditional formatting rules without restarting the server.
* **Control Room Interface:** A highly optimized, auto-refreshing Streamlit application that continuously polls the API for production logs with zero interface locking.

### 4. Production Line Simulator (`data_streamer.py`)
* **Data Streaming:** Acts as a continuous data pump, reading the master dataset iteratively and translating Numpy data types into pure JSON formats for HTTP POST requests.
* **Load Testing:** Simulates realistic factory constraints by injecting asynchronous payloads into the FastAPI endpoint.

## Installation & Setup

Ensure Python 3.9+ is installed, alongside a local instance of Ollama (for LLM reporting features) and the Graphviz engine (for tree visualization).

```bash
# Clone the repository
git clone [https://github.com/pervinturk/predictive-quality-os.git](https://github.com/pervinturk/predictive-quality-os.git)
cd predictive-quality-os

# Install dependencies
pip install -r requirements.txt
