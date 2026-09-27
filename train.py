import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Define feature names matching the 24 clinical parameters in main.py
num_cols = [
    "age", "bp", "sg", "al", "su", "bgr", "bu",
    "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"
]
cat_cols = [
    "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"
]

# 2. Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# 3. Combine preprocessing and Random Forest classifier into a single pipeline
model_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]
)

# 4. Sample training data (Replace with your actual dataset if loading from CSV)
# Example: df = pd.read_csv("kidney_disease.csv")
dummy_data = {
    "age": [52.0], "bp": [85.0], "sg": [1.015], "al": [1.0], "su": [0.0],
    "rbc": ["normal"], "pc": ["normal"], "pcc": ["notpresent"], "ba": ["notpresent"],
    "bgr": [145.0], "bu": [58.0], "sc": [2.4], "sod": [134.0], "pot": [4.8],
    "hemo": [10.2], "pcv": [32.0], "wc": [9200.0], "rc": [3.9],
    "htn": ["yes"], "dm": ["yes"], "cad": ["no"], "appet": ["poor"], "pe": ["no"], "ane": ["yes"]
}
X_train = pd.DataFrame(dummy_data)
y_train = [1]  # Target: 1 for CKD, 0 for Normal

# 5. Train model pipeline
model_pipeline.fit(X_train, y_train)

# 6. Save as 'kidney_model.pkl' expected by main.py
with open("kidney_model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

print("SUCCESS: 'kidney_model.pkl' trained and saved successfully.")