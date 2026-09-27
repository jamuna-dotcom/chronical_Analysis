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

## 🚀 Getting Started

### 1️⃣ Clone & Setup

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/kidneyai-pro.git](https://github.com/YOUR_USERNAME/kidneyai-pro.git)
cd kidneyai-pro

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install requirements
python -m pip install -r requirements.txt

```

### 2️⃣ Train & Export Model

Ensure `kidney_model.pkl` is built and present in the project directory:

```bash
python train.py

```

### 3️⃣ Launch Application Services

Open two terminal windows:

* **Terminal 1 (Backend API Service):**
```bash
uvicorn main:app --port 8000 --reload

```


* **Terminal 2 (Frontend Clinical Portal):**
```bash
streamlit run app.py

```



---

## 🔗 API Endpoints

The FastAPI microservice exposes structured JSON endpoints for seamless integration with external Electronic Health Record (EHR) systems:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | `GET` | Health check & model verification status |
| `/predict` | `POST` | Patient evaluation & SHAP explanation output |
| `/history` | `GET` | Historical patient audit logs |
| `/stats` | `GET` | Aggregated system metrics & CKD prevalence |
| `/generate-pdf` | `POST` | Dynamic clinical PDF report generation |

---

## 🛡️ Tech Stack

* **Frontend:** Streamlit, Plotly, HTML/CSS
* **Backend:** FastAPI, Pydantic, ReportLab (PDF Generation)
* **Machine Learning:** Scikit-Learn, SHAP, NumPy, Pandas
* **Database & Persistence:** SQLite3, Joblib

---

## ⚖️ Medical Disclaimer

> **IMPORTANT:** KidneyAI Pro is strictly designed as a decision-support and educational platform powered by machine learning models. All diagnostic risk assessments, staging suggestions, and generated medical reports must be validated by a certified healthcare professional or physician.

---

Made with ❤️ for Clinical Analytics & AI Healthcare
