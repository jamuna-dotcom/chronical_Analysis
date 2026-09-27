from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import pickle
import os

# --- FastAPI App Initialization ---
app = FastAPI(
    title="KidneyAI Clinical Decision Support API",
    description="REST API endpoint for Chronic Kidney Disease (CKD) machine learning predictions.",
    version="1.0.0"
)

MODEL_PATH = "kidney_model.pkl"

# Load ML Pipeline on startup
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Ensure main.py was executed.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# --- Input Schema Definition ---
class PatientData(BaseModel):
    age: float = Field(..., example=45.0)
    bp: float = Field(..., example=80.0)
    sg: float = Field(..., example=1.020)
    al: float = Field(..., example=0.0)
    su: float = Field(..., example=0.0)
    rbc: str = Field(..., example="normal")
    pc: str = Field(..., example="normal")
    pcc: str = Field(..., example="notpresent")
    ba: str = Field(..., example="notpresent")
    bgr: float = Field(..., example=120.0)
    bu: float = Field(..., example=40.0)
    sc: float = Field(..., example=1.2)
    sod: float = Field(..., example=140.0)
    pot: float = Field(..., example=4.5)
    hemo: float = Field(..., example=13.0)
    pcv: float = Field(..., example=40.0)
    wc: float = Field(..., example=8000.0)
    rc: float = Field(..., example=5.0)
    htn: str = Field(..., example="no")
    dm: str = Field(..., example="no")
    cad: str = Field(..., example="no")
    appet: str = Field(..., example="good")
    pe: str = Field(..., example="no")
    ane: str = Field(..., example="no")

# --- Endpoints ---
@app.get("/")
def health_check():
    return {"status": "online", "system": "KidneyAI API", "version": "1.0.0"}

@app.post("/predict")
def predict_ckd(patient: PatientData):
    try:
        # Convert Pydantic model to pandas DataFrame matching model inputs
        input_data = pd.DataFrame([patient.dict()])
        
        # Execute ML prediction
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        
        ckd_probability = float(probabilities[1] * 100)
        
        return {
            "status": "success",
            "prediction": prediction,
            "diagnosis": "High Risk - CKD Detected" if prediction == 1 else "Low Risk - Non-CKD",
            "ckd_probability_percent": round(ckd_probability, 2),
            "non_ckd_probability_percent": round(float(probabilities[0] * 100), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")