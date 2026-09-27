import io
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="KidneyAI Pro | Advanced Medical & Analytics Portal",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000"
# Inject High-Contrast Dark Theme CSS Fixes
st.markdown(
    """
    <style>
    /* Universal App Base Background & White Text */
    .stApp {
        background-color: #080d1a !important;
        color: #ffffff !important;
    }

    /* Target standard typography without breaking UI component internals */
    .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #ffffff !important;
    }

    /* Input Fields & Number Boxes */
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        background-color: #111c35 !important;
        color: #ffffff !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    /* Selectboxes */
    div[data-baseweb="select"] > div {
        background-color: #111c35 !important;
        color: #ffffff !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] span {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* ========================================================= */
    /* FILE UPLOADER COMPLETE DARK MODE & HIGH-CONTRAST FIX      */
    /* ========================================================= */
    
    /* Outer container override */
    section[data-testid="stFileUploader"] {
        background-color: transparent !important;
    }

    /* Dropzone box container override */
    div[data-testid="stFileUploaderDropzone"],
    div[data-baseweb="file-uploader"] {
        background-color: #0f172a !important;
        border: 2px dashed #0284c7 !important;
        border-radius: 12px !important;
        padding: 24px !important;
    }

    /* Internal background reset for BaseWeb child elements */
    div[data-testid="stFileUploaderDropzone"] > div,
    div[data-baseweb="file-uploader"] > div {
        background-color: #0f172a !important;
    }

    /* Inner File Uploader Button Fix */
    div[data-testid="stFileUploaderDropzone"] button,
    div[data-baseweb="file-uploader"] button {
        background-color: #0284c7 !important;
        border: 1px solid #0369a1 !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        padding: 6px 16px !important;
    }

    /* Hover effect for internal upload button */
    div[data-testid="stFileUploaderDropzone"] button:hover,
    div[data-baseweb="file-uploader"] button:hover {
        background-color: #0369a1 !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
    }

    /* Specificity fix for text labels inside dropzone (200MB limit text) */
    div[data-testid="stFileUploaderDropzone"] small,
    div[data-testid="stFileUploaderDropzone"] span,
    div[data-baseweb="file-uploader"] small,
    div[data-baseweb="file-uploader"] span {
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }

    /* Dataframe Table Dark High-Contrast Overrides */
    div[data-testid="stDataFrame"] {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }

    /* Metric KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.45);
        margin-bottom: 15px;
    }
    .kpi-title {
        color: #94a3b8 !important;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        margin-top: 6px;
        margin-bottom: 4px;
        color: #ffffff !important;
    }
    .kpi-sub {
        color: #64748b !important;
        font-size: 0.8rem;
    }

    /* Navigation Tabs */
    button[data-baseweb="tab"] p {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        font-size: 1.0rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #38bdf8 !important;
        font-weight: 800 !important;
    }

    /* Medical Disclaimer Footer */
    .disclaimer-box {
        background-color: #1e1b4b;
        border-left: 4px solid #6366f1;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    .disclaimer-text {
        color: #c7d2fe !important;
        font-size: 0.82rem;
        line-height: 1.35;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Backend Health Check
try:
    health_res = requests.get(f"{API_URL}/health", timeout=2)
    if health_res.status_code != 200 or not health_res.json().get("model_loaded"):
        st.error("⚠️ FastAPI Backend is online, but 'kidney_model.pkl' could not be loaded.")
        st.stop()
except Exception:
    st.error("❌ Cannot connect to FastAPI Backend. Start backend via `uvicorn main:app --port 8000`.")
    st.stop()

# ----------------------------------------------------
# GLOBAL CLINICAL EXECUTIVE DASHBOARD SUMMARY
# ----------------------------------------------------
stats_res = requests.get(f"{API_URL}/stats")
if stats_res.status_code == 200:
    g_stats = stats_res.json()
    st.markdown("##### 📊 Global System Diagnostic Overview")
    g1, g2, g3, g4, g5 = st.columns(5)
    
    with g1:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-title">Total Records</div>
                <div class="kpi-value">{g_stats['total_patients']}</div>
                <div class="kpi-sub">Total database evaluations</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with g2:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-title">CKD Positive</div>
                <div class="kpi-value" style="color:#ef4444 !important;">{g_stats['ckd_patients']}</div>
                <div class="kpi-sub">Identified risk profiles</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with g3:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-title">Normal Status</div>
                <div class="kpi-value" style="color:#22c55e !important;">{g_stats['normal_patients']}</div>
                <div class="kpi-sub">Healthy profiles</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with g4:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-title">CKD Prevalence</div>
                <div class="kpi-value">{g_stats['ckd_percentage']:.1f}%</div>
                <div class="kpi-sub">Prevalence rate</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with g5:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-title">Normal Rate</div>
                <div class="kpi-value">{g_stats['normal_percentage']:.1f}%</div>
                <div class="kpi-sub">Healthy rate</div>
            </div>""",
            unsafe_allow_html=True,
        )

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Patient Assessment & Diagnosis",
    "📂 Multi-Format File Upload & Automated EDA",
    "📜 Patient Historical Audit Log",
    "🔬 SHAP Local Diagnostic Explanations",
])

# ----------------------------------------------------
# TAB 1: SINGLE PATIENT DIAGNOSIS & PDF REPORT
# ----------------------------------------------------
with tab1:
    st.subheader("Patient Clinical Data Input")

    r1_1, r1_2 = st.columns(2)
    with r1_1:
        patient_id = st.text_input("Patient ID", value="PAT-8821")
    with r1_2:
        patient_name = st.text_input("Patient Name", value="Eleanor Vance")

    st.divider()
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("##### 🧍 Physical Vitals & Urine")
        age = st.number_input("Age", 1, 120, 52)
        bp = st.number_input("Blood Pressure (mmHg)", 40, 250, 85)
        sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=2)
        al = st.selectbox("Albumin Level", [0, 1, 2, 3, 4, 5], index=1)
        su = st.selectbox("Sugar Level", [0, 1, 2, 3, 4, 5], index=0)
        rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
        pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
        pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"])

    with c2:
        st.markdown("##### 🩸 Key Laboratory Biomarkers")
        ba = st.selectbox("Bacteria", ["notpresent", "present"])
        bgr = st.number_input("Blood Glucose Random (mg/dL)", 40.0, 500.0, 145.0)
        bu = st.number_input("Blood Urea (mg/dL)", 1.0, 300.0, 58.0)
        sc = st.number_input("Serum Creatinine (mg/dL)", 0.1, 20.0, 2.4)
        sod = st.number_input("Sodium (mEq/L)", 80.0, 200.0, 134.0)
        pot = st.number_input("Potassium (mEq/L)", 1.0, 15.0, 4.8)
        hemo = st.number_input("Hemoglobin (g/dL)", 3.0, 25.0, 10.2)
        pcv = st.number_input("Packed Cell Volume", 5.0, 70.0, 32.0)

    with c3:
        st.markdown("##### ❤️ Clinical History & Symptoms")
        wc = st.number_input("WBC Count", 1000.0, 30000.0, 9200.0)
        rc = st.number_input("RBC Count", 1.0, 10.0, 3.9)
        htn = st.selectbox("Hypertension", ["no", "yes"], index=1)
        dm = st.selectbox("Diabetes Mellitus", ["no", "yes"], index=1)
        cad = st.selectbox("Coronary Artery Disease", ["no", "yes"])
        appet = st.selectbox("Appetite", ["good", "poor"], index=1)
        pe = st.selectbox("Pedal Edema", ["no", "yes"])
        ane = st.selectbox("Anemia", ["no", "yes"], index=1)

    payload = {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "age": age, "bp": bp, "sg": sg, "al": al, "su": su,
        "rbc": rbc, "pc": pc, "pcc": pcc, "ba": ba, "bgr": bgr,
        "bu": bu, "sc": sc, "sod": sod, "pot": pot, "hemo": hemo,
        "pcv": pcv, "wc": wc, "rc": rc, "htn": htn, "dm": dm,
        "cad": cad, "appet": appet, "pe": pe, "ane": ane
    }

    st.divider()
    if st.button("🚀 Identify Disease & Evaluate Risk", type="primary"):
        res = requests.post(f"{API_URL}/predict", json=payload)

        if res.status_code == 200:
            data = res.json()
            st.session_state["active_result"] = data
            st.session_state["active_payload"] = payload
            st.success("Diagnostic Evaluation Completed Successfully.")
        else:
            st.error(f"Inference Engine Error: {res.text}")

    if "active_result" in st.session_state:
        res_data = st.session_state["active_result"]
        pred = res_data["prediction"]
        prob = res_data["ckd_probability"]
        status = res_data["risk_status"]
        color_hex = "#ef4444" if pred == 1 else "#22c55e"

        st.markdown("### 📋 Final Diagnostic Summary")
        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.markdown(
                f"""<div class="kpi-card">
                    <div class="kpi-title">Identified Outcome</div>
                    <div class="kpi-value" style="color:{color_hex} !important;">{status}</div>
                    <div class="kpi-sub">Patient: {res_data['patient_name']} ({res_data['patient_id']})</div>
                </div>""",
                unsafe_allow_html=True,
            )

        with kpi2:
            st.markdown(
                f"""<div class="kpi-card">
                    <div class="kpi-title">Model Confidence Score</div>
                    <div class="kpi-value">{prob:.2f}%</div>
                    <div class="kpi-sub">Probability of Chronic Kidney Disease</div>
                </div>""",
                unsafe_allow_html=True,
            )

        with kpi3:
            st.markdown(
                f"""<div class="kpi-card">
                    <div class="kpi-title">Key Biomarker Status</div>
                    <div class="kpi-value">{sc:.2f} mg/dL</div>
                    <div class="kpi-sub">Serum Creatinine · Hgb: {hemo:.2f} g/dL</div>
                </div>""",
                unsafe_allow_html=True,
            )

        pdf_res = requests.post(
            f"{API_URL}/generate-pdf",
            params={"prediction": pred, "ckd_probability": prob},
            json=st.session_state["active_payload"],
        )

        if pdf_res.status_code == 200:
            st.download_button(
                label="📄 Download Official PDF Diagnostic Report",
                data=pdf_res.content,
                file_name=f"KidneyAI_Report_{res_data['patient_id']}.pdf",
                mime="application/pdf",
            )

# ----------------------------------------------------
# TAB 2: MULTI-FORMAT FILE UPLOAD & AUTOMATED ANALYTICS
# ----------------------------------------------------
with tab2:
    st.subheader("Universal Data Ingestion & Automated Chart Analytics Engine")
    st.markdown(
        "Upload dataset files in **CSV**, **Excel (.xlsx, .xls)**, **JSON**, or **Parquet** format for data parsing, automated chart generation, and machine learning inference."
    )

    uploaded_file = st.file_uploader(
        "Choose Patient or Clinical Data File",
        type=["csv", "xlsx", "xls", "json", "parquet"],
    )

    if uploaded_file is not None:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        df = None

        try:
            if file_ext == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_ext in ["xlsx", "xls"]:
                df = pd.read_excel(uploaded_file)
            elif file_ext == "json":
                df = pd.read_json(uploaded_file)
            elif file_ext == "parquet":
                df = pd.read_parquet(uploaded_file)
        except Exception as read_err:
            st.error(f"❌ Failed to parse uploaded file: {read_err}")

        if df is not None and not df.empty:
            st.success(f"✅ Successfully ingested file: `{uploaded_file.name}` ({len(df)} rows, {len(df.columns)} columns)")

            # DATA PREVIEW SECTION
            st.markdown("### 🔍 Uploaded Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # DATASET SUMMARY STATS
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Records / Rows", len(df))
            with m2:
                st.metric("Total Parameters / Cols", len(df.columns))
            with m3:
                st.metric("Numeric Biomarkers", len(df.select_dtypes(include=['number']).columns))
            with m4:
                st.metric("Categorical Features", len(df.select_dtypes(include=['object', 'category']).columns))

            st.divider()

            # AUTOMATED SMART CHART GENERATION
            st.markdown("### 📊 Automated Visual Analytics & Chart Generation")
            num_cols = list(df.select_dtypes(include=['number']).columns)
            cat_cols = list(df.select_dtypes(include=['object', 'category', 'bool']).columns)

            generated_charts = []

            c_col1, c_col2 = st.columns(2)

            # Chart 1: Numerical Distributions or Top Feature Histogram
            with c_col1:
                if len(num_cols) >= 1:
                    primary_num = num_cols[0]
                    fig1 = px.histogram(
                        df, x=primary_num, title=f"Distribution Analysis: {primary_num}",
                        color_discrete_sequence=["#38bdf8"], marginal="box"
                    )
                    fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#ffffff"})
                    st.plotly_chart(fig1, use_container_width=True)
                    generated_charts.append(("Distribution_Chart.png", fig1))
                else:
                    st.info("No numerical parameters available for histogram distribution.")

            # Chart 2: Categorical Breakdown or Value Counts
            with c_col2:
                if len(cat_cols) >= 1:
                    primary_cat = cat_cols[0]
                    counts = df[primary_cat].value_counts().reset_index()
                    counts.columns = [primary_cat, "Count"]
                    fig2 = px.pie(
                        counts, names=primary_cat, values="Count",
                        title=f"Categorical Composition: {primary_cat}",
                        hole=0.4, color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#ffffff"})
                    st.plotly_chart(fig2, use_container_width=True)
                    generated_charts.append(("Categorical_Composition.png", fig2))
                elif len(num_cols) >= 2:
                    fig2 = px.scatter(
                        df, x=num_cols[0], y=num_cols[1],
                        title=f"Correlation Scatter: {num_cols[0]} vs {num_cols[1]}",
                        color_discrete_sequence=["#22c55e"]
                    )
                    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#ffffff"})
                    st.plotly_chart(fig2, use_container_width=True)
                    generated_charts.append(("Scatter_Correlation.png", fig2))

            # Chart 3: Correlation Matrix Heatmap if Multiple Numeric Features Exist
            if len(num_cols) >= 3:
                st.markdown("##### Biomarker Correlation Matrix")
                corr = df[num_cols].corr()
                fig_corr = px.imshow(
                    corr, text_auto=True, color_continuous_scale="Viridis",
                    title="Biomarker Linear Correlation Heatmap"
                )
                fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#ffffff"})
                st.plotly_chart(fig_corr, use_container_width=True)
                generated_charts.append(("Correlation_Heatmap.png", fig_corr))

            st.divider()

            # CHECK FOR CKD PREDICTION COMPATIBILITY
            required_ckd_features = [
                "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
                "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
                "htn", "dm", "cad", "appet", "pe", "ane"
            ]

            missing_ckd_cols = [col for col in required_ckd_features if col not in df.columns]

            if not missing_ckd_cols:
                st.success("🤖 Disease Inference Compatibility Detected: All 24 CKD parameters present.")

                if st.button("⚡ Execute Model Batch Prediction on Uploaded File", type="primary"):
                    # Clean DataFrame missing values before serializing to JSON
                    df_clean = df.replace({np.nan: None})
                    records = df_clean.to_dict(orient="records")
                    
                    with st.spinner("Processing predictions via backend..."):
                        try:
                            batch_res = requests.post(f"{API_URL}/predict-batch", json=records)

                            if batch_res.status_code == 200:
                                pred_results = batch_res.json()
                                pred_df = pd.DataFrame(pred_results)
                                st.markdown("### 📋 Predictive Disease Diagnostic Output")
                                st.dataframe(pred_df, use_container_width=True)

                                # Download Analyzed Data CSV
                                csv_data = pred_df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    label="📥 Download Analyzed Predictive Data (CSV)",
                                    data=csv_data,
                                    file_name="Analyzed_CKD_Predictions.csv",
                                    mime="text/csv",
                                )
                            else:
                                st.error(f"Inference failure: {batch_res.text}")
                        except Exception as req_err:
                            st.error(f"Request failed: {req_err}")
            else:
                st.info(f"ℹ️ Generic Dataset Detected: File processed for exploratory analytics. (Missing CKD model parameters: {missing_ckd_cols[:4]}...)")

                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Processed Dataset (CSV)",
                    data=csv_data,
                    file_name="Processed_Dataset.csv",
                    mime="text/csv",
                )

# ----------------------------------------------------
# TAB 3: SEARCHABLE HISTORICAL AUDIT TRAIL
# ----------------------------------------------------
with tab3:
    st.subheader("Patient Audit Log & Searchable Historical Records")

    search_query = st.text_input("🔍 Search Historical Records by Patient ID or Patient Name", value="")

    hist_params = {}
    if search_query.strip():
        hist_params["search"] = search_query.strip()

    hist_res = requests.get(f"{API_URL}/history", params=hist_params)

    if hist_res.status_code == 200:
        records = hist_res.json()
        if not records:
            st.info("No patient records found matching your search query.")
        else:
            hist_df = pd.DataFrame(records)

            st.markdown(f"##### Showing {len(hist_df)} Patient Diagnostic Audit Logs")
            st.dataframe(hist_df, use_container_width=True)

            if len(hist_df) > 1 and "timestamp" in hist_df.columns:
                st.markdown("##### Risk Progression Trend Analysis")
                hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])

                fig_trend = px.line(
                    hist_df.sort_values("timestamp"),
                    x="timestamp",
                    y="ckd_probability",
                    color="patient_name",
                    markers=True,
                    title="CKD Risk Progression Trajectory Across Evaluations",
                )
                fig_trend.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#ffffff"},
                )
                st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.error("Failed to load historical records from backend server.")

# ----------------------------------------------------
# TAB 4: SHAP EXPLANATIONS
# ----------------------------------------------------
with tab4:
    st.subheader("Local Patient Biomarker Contribution (SHAP)")
    if "active_result" not in st.session_state:
        st.info("Run a single-patient evaluation in Tab 1 to compute SHAP diagnostic impact values.")
    else:
        active_res = st.session_state["active_result"]
        shap_data = active_res.get("shap_contributions", {})

        if "info" in shap_data:
            st.warning(shap_data["info"])
        else:
            shap_df = (
                pd.DataFrame(list(shap_data.items()), columns=["Biomarker Feature", "SHAP Impact Value"])
                .sort_values(by="SHAP Impact Value", key=abs, ascending=True)
                .tail(12)
            )

            shap_df["Impact Category"] = shap_df["SHAP Impact Value"].apply(
                lambda val: "Elevates Disease Risk" if val > 0 else "Lowers Disease Risk"
            )

            fig_shap = px.bar(
                shap_df,
                x="SHAP Impact Value",
                y="Biomarker Feature",
                color="Impact Category",
                orientation="h",
                color_discrete_map={
                    "Elevates Disease Risk": "#ef4444",
                    "Lowers Disease Risk": "#22c55e",
                },
                title=f"Feature Significance Breakdown for {active_res['patient_name']}",
            )
            fig_shap.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#ffffff"},
                height=480,
            )
            st.plotly_chart(fig_shap, use_container_width=True)

# Permanent Medical Disclaimer Footer
st.markdown(
    """
    <div class="disclaimer-box">
        <p class="disclaimer-text">
            <b>MEDICAL DISCLAIMER:</b> This platform is designed solely as a clinical decision support tool powered by machine learning algorithms. 
            All diagnostic predictions, risk scores, and generated reports must be reviewed by a licensed physician or clinical professional. 
            This system does not replace professional medical diagnosis or clinical judgment.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)