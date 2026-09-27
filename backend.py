from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pickle
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="KidneyAI Backend",
    description="Backend API for Chronic Kidney Disease Prediction",
    version="1.0.0"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "kidney_model.pkl"
)

DATABASE_PATH = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    "kidneyai.db"
)


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

try:

    with open(MODEL_PATH, "rb") as file:

        model = pickle.load(file)

    MODEL_STATUS = "online"

except Exception as error:

    model = None

    MODEL_STATUS = "offline"

    MODEL_ERROR = str(error)


# ============================================================
# DATABASE
# ============================================================

def create_database():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            age INTEGER,

            bp REAL,

            sg REAL,

            al INTEGER,

            su INTEGER,

            bgr REAL,

            bu REAL,

            sc REAL,

            sod REAL,

            pot REAL,

            hemo REAL,

            pcv REAL,

            wc REAL,

            rc REAL,

            rbc TEXT,

            pc TEXT,

            pcc TEXT,

            ba TEXT,

            htn TEXT,

            dm TEXT,

            cad TEXT,

            appet TEXT,

            pe TEXT,

            ane TEXT,

            prediction TEXT,

            confidence REAL,

            created_at TEXT
        )
        """
    )

    connection.commit()

    connection.close()


create_database()


# ============================================================
# REQUEST DATA MODEL
# ============================================================

class PatientData(BaseModel):

    age: int

    bp: float

    sg: float

    al: int

    su: int

    rbc: str

    pc: str

    pcc: str

    ba: str

    bgr: float

    bu: float

    sc: float

    sod: float

    pot: float

    hemo: float

    pcv: float

    wc: float

    rc: float

    htn: str

    dm: str

    cad: str

    appet: str

    pe: str

    ane: str


# ============================================================
# ROOT API
# ============================================================

@app.get("/")
def home():

    return {

        "application": "KidneyAI",

        "message":
            "Chronic Kidney Disease Prediction API",

        "status": "running",

        "model": MODEL_STATUS
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "healthy",

        "model_status": MODEL_STATUS,

        "timestamp":
            datetime.now().isoformat()

    }


# ============================================================
# PREPROCESSING
# ============================================================

def prepare_input(patient: PatientData):

    data = {

        "age": patient.age,

        "bp": patient.bp,

        "sg": patient.sg,

        "al": patient.al,

        "su": patient.su,

        "rbc": patient.rbc,

        "pc": patient.pc,

        "pcc": patient.pcc,

        "ba": patient.ba,

        "bgr": patient.bgr,

        "bu": patient.bu,

        "sc": patient.sc,

        "sod": patient.sod,

        "pot": patient.pot,

        "hemo": patient.hemo,

        "pcv": patient.pcv,

        "wc": patient.wc,

        "rc": patient.rc,

        "htn": patient.htn,

        "dm": patient.dm,

        "cad": patient.cad,

        "appet": patient.appet,

        "pe": patient.pe,

        "ane": patient.ane
    }

    df = pd.DataFrame([data])


    # Common CKD dataset encoding

    encoding = {

        "normal": 1,

        "abnormal": 0,

        "present": 1,

        "notpresent": 0,

        "yes": 1,

        "no": 0,

        "good": 1,

        "poor": 0
    }


    categorical_columns = [

        "rbc",

        "pc",

        "pcc",

        "ba",

        "htn",

        "dm",

        "cad",

        "appet",

        "pe",

        "ane"

    ]


    for column in categorical_columns:

        df[column] = (
            df[column]
            .map(encoding)
            .fillna(0)
        )


    return df


# ============================================================
# PREDICTION API
# ============================================================

@app.post("/predict")
def predict(patient: PatientData):

    if model is None:

        raise HTTPException(

            status_code=500,

            detail=
                "Machine learning model could not be loaded."
        )


    try:

        input_data = prepare_input(
            patient
        )


        # ----------------------------------------
        # PREDICTION
        # ----------------------------------------

        prediction = model.predict(
            input_data
        )


        result = str(
            prediction[0]
        )


        # ----------------------------------------
        # NORMALIZE RESULT
        # ----------------------------------------

        result_lower = result.lower()


        if (

            result_lower in [
                "1",
                "ckd",
                "yes",
                "positive"
            ]

            or

            "ckd" in result_lower

        ):

            prediction_text = "CKD"

        else:

            prediction_text = "Not CKD"


        # ----------------------------------------
        # CONFIDENCE
        # ----------------------------------------

        confidence = None


        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model.predict_proba(
                    input_data
                )
            )

            confidence = float(
                np.max(
                    probabilities[0]
                ) * 100
            )


        # ----------------------------------------
        # SAVE TO DATABASE
        # ----------------------------------------

        save_prediction(

            patient,

            prediction_text,

            confidence

        )


        return {

            "success": True,

            "prediction":
                prediction_text,

            "confidence":
                confidence,

            "timestamp":
                datetime.now().isoformat()

        }


    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error)

        )


# ============================================================
# SAVE PREDICTION
# ============================================================

def save_prediction(

    patient,

    prediction,

    confidence

):

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()


    cursor.execute(

        """
        INSERT INTO predictions (

            age,
            bp,
            sg,
            al,
            su,

            bgr,
            bu,
            sc,
            sod,
            pot,

            hemo,
            pcv,
            wc,
            rc,

            rbc,
            pc,
            pcc,
            ba,

            htn,
            dm,
            cad,
            appet,
            pe,
            ane,

            prediction,
            confidence,
            created_at

        )

        VALUES (

            ?, ?, ?, ?, ?,

            ?, ?, ?, ?, ?,

            ?, ?, ?, ?,

            ?, ?, ?, ?,

            ?, ?, ?, ?, ?, ?,

            ?, ?, ?

        )
        """,

        (

            patient.age,

            patient.bp,

            patient.sg,

            patient.al,

            patient.su,


            patient.bgr,

            patient.bu,

            patient.sc,

            patient.sod,

            patient.pot,


            patient.hemo,

            patient.pcv,

            patient.wc,

            patient.rc,


            patient.rbc,

            patient.pc,

            patient.pcc,

            patient.ba,


            patient.htn,

            patient.dm,

            patient.cad,

            patient.appet,

            patient.pe,

            patient.ane,


            prediction,

            confidence,

            datetime.now().isoformat()

        )

    )


    connection.commit()

    connection.close()


# ============================================================
# GET PREDICTION HISTORY
# ============================================================

@app.get("/predictions")
def get_predictions():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    dataframe = pd.read_sql_query(

        """
        SELECT *
        FROM predictions
        ORDER BY id DESC
        """,

        connection

    )

    connection.close()


    return dataframe.to_dict(
        orient="records"
    )


# ============================================================
# GET STATISTICS
# ============================================================

@app.get("/statistics")
def statistics():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        """
    )

    total = cursor.fetchone()[0]


    cursor.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        WHERE prediction = 'CKD'
        """
    )

    ckd = cursor.fetchone()[0]


    cursor.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        WHERE prediction = 'Not CKD'
        """
    )

    not_ckd = cursor.fetchone()[0]


    connection.close()


    return {

        "total_predictions":
            total,

        "ckd_predictions":
            ckd,

        "not_ckd_predictions":
            not_ckd

    }