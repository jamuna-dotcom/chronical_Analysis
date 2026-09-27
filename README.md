# 🩺 KidneyAI Pro — Advanced Clinical Analytics & Diagnostic Platform

KidneyAI Pro is a full-stack, enterprise-grade decision support platform designed to assist healthcare professionals in identifying Chronic Kidney Disease (CKD) risk, calculating KDIGO clinical stages, and exploring feature-level AI explanations (SHAP) in real time.

---

## 🌟 Key Features

* **Real-Time ML Inference Engine:** Machine Learning pipeline powered by Scikit-Learn and FastAPI.
* **Interactive Clinical Portal:** Built with Streamlit, featuring high-contrast dark theme UI components.
* **KDIGO eGFR Staging:** Automatic calculation of Glomerular Filtration Rate (CKD-EPI equation) mapped to official clinical stages ($G1$ to $G5$).
* **Real-Time 'What-If' Simulator:** Dynamic parameter sliders allowing clinicians to simulate medical treatment scenarios and view live risk deltas.
* **SHAP Diagnostic Interpretability:** Model explanation capabilities to highlight biomarker contributions.
* **Comparative Patient Matrix:** Side-by-side radar overlay charts for evaluating multi-patient clinical histories.
* **Universal Data Ingestion & Automated EDA:** Multi-format parser (`.csv`, `.xlsx`, `.json`, `.parquet`) for exploratory data visualization.

---

## 🏗️ Architecture & Project Structure

```text
kidneyai-pro/
├── app.py                # Streamlit Frontend Web Dashboard
├── main.py               # FastAPI RESTful Microservice API
├── train.py              # Machine Learning Training & Pipeline Export
├── kidney_model.pkl      # Trained Pipeline Model Binary
├── requirements.txt      # Project Dependencies
├── Dockerfile            # Container Deployment Spec
├── docker-compose.yml    # Multi-container Orchestration
└── README.md             # Documentation
