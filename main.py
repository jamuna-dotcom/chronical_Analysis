import io
import importlib
import os
import pickle
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

# ReportLab for PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

app = FastAPI(
    title="KidneyAI Pro - Clinical Decision Support Platform API",
    description="Backend microservice for Chronic Kidney Disease identification, patient historical tracking, multi-format batch evaluation, and PDF reports.",
    version="3.1.0",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "kidney_model.pkl"
DB_PATH = BASE_DIR / "patient_history.db"

model_pipeline = None


def initialize_database() -> None:
    """Initialize SQLite database for patient history persistence."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            patient_name TEXT,
            age REAL,
            bp REAL,
            sc REAL,
            hemo REAL,
            bu REAL,
            bgr REAL,
            prediction INTEGER,
            ckd_probability REAL,
            risk_status TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    existing_columns = {
        row[1] for row in cursor.execute("PRAGMA table_info(predictions)")
    }
    if "risk_status" not in existing_columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN risk_status TEXT")
    conn.commit()
    conn.close()


def load_model():
    global model_pipeline
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model file missing: {MODEL_PATH}")
        return False
    try:
        # Older artifacts may reference NumPy's private module path.
        import numpy.core
        import numpy.core.numeric
        import sklearn

        sys.modules.setdefault("numpy._core", numpy.core)
        sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)

        with open(MODEL_PATH, "rb") as f:
            model_pipeline = pickle.load(f)
        print(f"[SUCCESS] Model loaded successfully from {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load pickle file: {e}")
        return False


initialize_database()
load_model()


class PatientInput(BaseModel):
    patient_id: str = Field(default="PAT-8821")
    patient_name: str = Field(default="Eleanor Vance")
    age: float = Field(default=52.0)
    bp: float = Field(default=85.0)
    sg: float = Field(default=1.015)
    al: float = Field(default=1.0)
    su: float = Field(default=0.0)
    rbc: str = Field(default="normal")
    pc: str = Field(default="normal")
    pcc: str = Field(default="notpresent")
    ba: str = Field(default="notpresent")
    bgr: float = Field(default=145.0)
    bu: float = Field(default=58.0)
    sc: float = Field(default=2.4)
    sod: float = Field(default=134.0)
    pot: float = Field(default=4.8)
    hemo: float = Field(default=10.2)
    pcv: float = Field(default=32.0)
    wc: float = Field(default=9200.0)
    rc: float = Field(default=3.9)
    htn: str = Field(default="yes")
    dm: str = Field(default="yes")
    cad: str = Field(default="no")
    appet: str = Field(default="poor")
    pe: str = Field(default="no")
    ane: str = Field(default="yes")


@app.get("/health")
def health_check():
    if model_pipeline is None:
        load_model()
    return {
        "status": "ok" if model_pipeline is not None else "degraded",
        "model_loaded": model_pipeline is not None,
    }


@app.get("/stats")
def get_global_stats():
    """Retrieve database metrics for the main dashboard overview."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_records = cursor.fetchone()[0] or 0

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 1")
    ckd_count = cursor.fetchone()[0] or 0

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 0")
    normal_count = cursor.fetchone()[0] or 0

    conn.close()

    ckd_pct = (ckd_count / total_records * 100) if total_records > 0 else 0.0
    normal_pct = (normal_count / total_records * 100) if total_records > 0 else 0.0

    return {
        "total_patients": total_records,
        "ckd_patients": ckd_count,
        "normal_patients": normal_count,
        "ckd_percentage": round(ckd_pct, 2),
        "normal_percentage": round(normal_pct, 2),
    }


@app.post("/predict")
def predict_ckd(patient: PatientInput):
    if model_pipeline is None:
        raise HTTPException(
            status_code=500, detail="Model 'kidney_model.pkl' is not loaded on backend server."
        )

    patient_dict = patient.dict()
    patient_id = str(patient_dict.pop("patient_id"))
    patient_name = str(patient_dict.pop("patient_name"))

    input_df = pd.DataFrame([patient_dict])

    try:
        prediction = int(model_pipeline.predict(input_df)[0])
        probabilities = model_pipeline.predict_proba(input_df)[0]
        classes = getattr(model_pipeline, "classes_", np.unique([prediction]))
        ckd_index = np.flatnonzero(np.asarray(classes) == 1)
        ckd_prob = float(probabilities[ckd_index[0]] * 100) if ckd_index.size else 0.0
        risk_status = "Chronic Kidney Disease (CKD) Detected" if prediction == 1 else "Non-CKD (Normal Risk Profile)"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference Failure: {str(e)}")

    # Save execution to historical database
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """
            INSERT INTO predictions (
                patient_id, patient_name, age, bp, sc, hemo, bu, bgr,
                prediction, ckd_probability, risk_status, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                patient_id,
                patient_name,
                float(patient.age),
                float(patient.bp),
                float(patient.sc),
                float(patient.hemo),
                float(patient.bu),
                float(patient.bgr),
                prediction,
                ckd_prob,
                risk_status,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as db_err:
        print(f"[DATABASE ERROR] Could not save history log: {db_err}")

    # Compute SHAP feature impacts
    shap_contributions = {}
    try:
        shap = importlib.import_module("shap")

        if hasattr(model_pipeline, "named_steps"):
            classifier = model_pipeline.named_steps["model"]
            preprocessor = model_pipeline.named_steps["preprocessing"]

            transformed_X = preprocessor.transform(input_df)
            num_features = list(preprocessor.transformers_[0][2])
            cat_features = list(
                preprocessor.transformers_[1][1]
                .named_steps["encoder"]
                .get_feature_names_out(preprocessor.transformers_[1][2])
            )
            all_feature_names = num_features + cat_features

            explainer = shap.TreeExplainer(classifier)
            shap_vals = explainer.shap_values(transformed_X)

            if isinstance(shap_vals, list):
                target_shap = shap_vals[1][0]
            elif len(shap_vals.shape) == 3:
                target_shap = shap_vals[0, :, 1]
            else:
                target_shap = shap_vals[0]

            shap_contributions = {
                feat: float(val) for feat, val in zip(all_feature_names, target_shap)
            }
        else:
            explainer = shap.Explainer(model_pipeline)
            shap_vals = explainer(input_df)
            shap_contributions = {
                feat: float(val) for feat, val in zip(input_df.columns, shap_vals.values[0])
            }
    except Exception as shap_err:
        shap_contributions = {"info": f"SHAP calculation skipped: {str(shap_err)}"}

    return {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "prediction": prediction,
        "ckd_probability": round(ckd_prob, 2),
        "risk_status": risk_status,
        "shap_contributions": shap_contributions,
    }


@app.post("/predict-batch")
def predict_batch_ckd(records: List[dict]):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model file not loaded.")

    if not records:
        raise HTTPException(status_code=400, detail="Input record list is empty.")

    input_df = pd.DataFrame(records)

    required_features = [
        "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
        "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
        "htn", "dm", "cad", "appet", "pe", "ane"
    ]

    missing_cols = [col for col in required_features if col not in input_df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Validation failed. Missing required feature columns: {missing_cols}"
        )

    try:
        feature_df = input_df[required_features].copy()
        predictions = model_pipeline.predict(feature_df)
        probability_matrix = model_pipeline.predict_proba(feature_df)
        classes = getattr(model_pipeline, "classes_", np.unique(predictions))
        ckd_index = np.flatnonzero(np.asarray(classes) == 1)
        if ckd_index.size:
            probabilities = probability_matrix[:, ckd_index[0]]
        else:
            probabilities = np.zeros(len(predictions))

        input_df["Prediction"] = predictions
        input_df["Disease_Status"] = [
            "Chronic Kidney Disease (CKD)" if p == 1 else "Normal (Non-CKD)" for p in predictions
        ]
        input_df["CKD_Probability_%"] = np.round(probabilities * 100, 2)

        return input_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch evaluation error: {str(e)}")


@app.get("/history")
def get_patient_history(search: Optional[str] = None):
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM predictions"
    params = []

    if search:
        query += " WHERE patient_id LIKE ? OR patient_name LIKE ?"
        term = f"%{search.strip()}%"
        params.extend([term, term])

    query += " ORDER BY timestamp DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    return df.to_dict(orient="records")


@app.post("/generate-pdf")
def generate_pdf_endpoint(patient: PatientInput, prediction: int, ckd_probability: float):
    if not HAS_REPORTLAB:
        raise HTTPException(
            status_code=500, detail="ReportLab library not installed on API server."
        )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=20, textColor=colors.HexColor("#0ea5e9"), spaceAfter=12)
    body_style = ParagraphStyle("BodyStyle", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#334155"), leading=14)
    disclaimer_style = ParagraphStyle("DisclaimerStyle", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#64748b"), leading=10)

    title = Paragraph("KidneyAI Pro - Clinical Diagnostic Decision Support Report", title_style)
    timestamp = Paragraph(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style)

    status_text = "Chronic Kidney Disease Detected" if prediction == 1 else "Non-CKD (Normal Risk)"
    status_color = "#ef4444" if prediction == 1 else "#22c55e"

    summary = Paragraph(
        f"<b>Patient Name:</b> {patient.patient_name}<br/>"
        f"<b>Patient ID:</b> {patient.patient_id}<br/>"
        f"<b>Diagnosis Result:</b> <font color='{status_color}'><b>{status_text}</b></font><br/>"
        f"<b>CKD Probability Confidence:</b> {ckd_probability:.2f}%",
        body_style,
    )

    table_data = [
        ["Biomarker / Feature", "Observed Value"],
        ["Age", f"{patient.age} yrs"],
        ["Blood Pressure", f"{patient.bp} mmHg"],
        ["Serum Creatinine", f"{patient.sc} mg/dL"],
        ["Blood Urea", f"{patient.bu} mg/dL"],
        ["Hemoglobin", f"{patient.hemo} g/dL"],
        ["Blood Glucose Random", f"{patient.bgr} mg/dL"],
        ["Specific Gravity", f"{patient.sg}"],
        ["Albumin Level", f"{patient.al}"],
        ["Hypertension", patient.htn.capitalize()],
        ["Diabetes Mellitus", patient.dm.capitalize()],
    ]

    table = Table(table_data, colWidths=[220, 260])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.HexColor("#f1f5f9")]),
            ]
        )
    )

    disclaimer = Paragraph(
        "<b>MEDICAL DISCLAIMER:</b> This report is generated by an artificial intelligence decision support system. "
        "It is designed solely to assist licensed healthcare professionals and MUST NOT replace professional medical diagnosis, "
        "clinical evaluation, or direct physician assessment.",
        disclaimer_style,
    )

    elements = [title, timestamp, Spacer(1, 10), summary, Spacer(1, 15), table, Spacer(1, 20), disclaimer]
    doc.build(elements)

    pdf_value = buffer.getvalue()
    buffer.close()

    return Response(
        content=pdf_value,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=Report_{patient.patient_id}.pdf"},
    )