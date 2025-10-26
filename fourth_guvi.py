# ==========================================
# Step 7: Multiple Disease Prediction System (Updated & Robust)
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Multiple Disease Prediction", layout="centered")

# ------------------------------
# Paths
# ------------------------------
MODEL_DIR = r"F:/GUVI/guvi project 4/saved_models"
DATA_DIR = r"F:/GUVI/guvi project 4/cleaned_data"

KIDNEY_CSV = os.path.join(DATA_DIR, "kidney_cleaned.csv")
PARKINSON_CSV = os.path.join(DATA_DIR, "parkinson_cleaned.csv")
LIVER_CSV = os.path.join(DATA_DIR, "liver_cleaned.csv")

# ------------------------------
# Helper Functions
# ------------------------------
def load_model_and_scaler(model_path, scaler_path):
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    st.warning(f"Missing model or scaler: {model_path}, {scaler_path}")
    return None, None

def get_feature_order(csv_path, target_cols=None):
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    for c in target_cols or []:
        if c in cols:
            cols.remove(c)
    return cols

def encode_kidney_row(row):
    r = row.copy()
    r['rbc'] = 1 if str(r['rbc']).lower() == 'normal' else 0
    r['pc'] = 1 if str(r['pc']).lower() == 'normal' else 0
    r['appet'] = 1 if str(r['appet']).lower() == 'good' else 0
    for c in ['pcc','ba','htn','dm','cad','pe','ane']:
        r[c] = 1 if str(r[c]).lower() in ['yes','1','true'] else 0
    return r

def apply_scaler_and_order(df_input, scaler, feature_order):
    for c in feature_order:
        if c not in df_input.columns:
            df_input[c] = 0
    df_input = df_input[feature_order].astype(float)
    return scaler.transform(df_input)

def get_prediction_label(model, X, disease_name):
    """Direct prediction without threshold"""
    pred = model.predict(X)[0]
    label = f"✅ {disease_name}" if pred == 1 else f"❌ No {disease_name}"
    return label

def get_probability_label(model, X, disease_name, threshold):
    """Prediction with threshold"""
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:,1][0]
    else:
        prob = model.predict(X)[0]
    label = f"✅ {disease_name}" if prob >= threshold else f"❌ No {disease_name}"
    return label, prob

# ------------------------------
# Load Models & Scalers
# ------------------------------
kidney_model, kidney_scaler = load_model_and_scaler(
    os.path.join(MODEL_DIR, "kidney_model.pkl"),
    os.path.join(MODEL_DIR, "kidney_scaler.pkl")
)
parkinson_model, parkinson_scaler = load_model_and_scaler(
    os.path.join(MODEL_DIR, "parkinson_model.pkl"),
    os.path.join(MODEL_DIR, "parkinson_scaler.pkl")
)
liver_model, liver_scaler = load_model_and_scaler(
    os.path.join(MODEL_DIR, "liver_model.pkl"),
    os.path.join(MODEL_DIR, "liver_scaler.pkl")
)

# ------------------------------
# Load feature orders
# ------------------------------
k_features = get_feature_order(KIDNEY_CSV, target_cols=['CKD','id'])
p_features = get_feature_order(PARKINSON_CSV, target_cols=['status'])
l_features = get_feature_order(LIVER_CSV, target_cols=['LiverDisease'])

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🩺 Multiple Disease Prediction System")
disease = st.selectbox("Select Disease", ["Kidney Disease", "Parkinson's Disease", "Liver Disease"])

# ------------------------------
# Parkinson's Disease Input
# ------------------------------
if disease == "Parkinson's Disease" and parkinson_model and parkinson_scaler:
    st.subheader("Parkinson’s Disease Input")
    input_data = {f: st.number_input(f, value=0.0) for f in p_features}
    threshold = st.slider("Set Prediction Threshold for Parkinson", 0.0, 1.0, 0.45, 0.05)
    if st.button("Predict Parkinson's"):
        X = apply_scaler_and_order(pd.DataFrame([input_data]), parkinson_scaler, p_features)
        label, prob = get_probability_label(parkinson_model, X, "Parkinson’s Disease", threshold)
        st.success(f"Prediction: {label} (Prob: {prob:.3f}, Threshold: {threshold})")

# ------------------------------
# Kidney Disease Input
# ------------------------------
elif disease == "Kidney Disease" and kidney_model and kidney_scaler:
    st.subheader("Kidney Disease Input")
    input_data = {
        'age': st.number_input("Age", 1, 100, 45),
        'bp': st.number_input("Blood Pressure", 50, 180, 80),
        'sg': st.number_input("Specific Gravity", 1.000, 1.030, 1.020),
        'al': st.number_input("Albumin", 0, 5, 0),
        'su': st.number_input("Sugar", 0, 5, 0),
        'rbc': st.selectbox("Red Blood Cells", ["normal","abnormal"]),
        'pc': st.selectbox("Pus Cell", ["normal","abnormal"]),
        'pcc': st.selectbox("Pus Cell Clumps", ["no","yes"]),
        'ba': st.selectbox("Bacteria", ["no","yes"]),
        'bgr': st.number_input("Blood Glucose Random", 50, 500, 120),
        'bu': st.number_input("Blood Urea", 5, 200, 40),
        'sc': st.number_input("Serum Creatinine", 0.1, 10.0, 1.0),
        'sod': st.number_input("Sodium", 100, 200, 140),
        'pot': st.number_input("Potassium", 2.0, 10.0, 4.5),
        'hemo': st.number_input("Hemoglobin", 3.0, 17.0, 13.0),
        'pcv': st.number_input("Packed Cell Volume", 10, 60, 40),
        'wc': st.number_input("White Blood Cell Count", 3000, 20000, 8000),
        'rc': st.number_input("Red Blood Cell Count", 2.0, 7.0, 5.0),
        'htn': st.selectbox("Hypertension", ["no","yes"]),
        'dm': st.selectbox("Diabetes Mellitus", ["no","yes"]),
        'cad': st.selectbox("Coronary Artery Disease", ["no","yes"]),
        'appet': st.selectbox("Appetite", ["good","poor"]),
        'pe': st.selectbox("Pedal Edema", ["no","yes"]),
        'ane': st.selectbox("Anemia", ["no","yes"])
    }
    threshold = st.slider("Set Prediction Threshold for CKD", 0.0, 1.0, 0.65, 0.05)
    if st.button("Predict Kidney"):
        encoded = encode_kidney_row(input_data)
        X = apply_scaler_and_order(pd.DataFrame([encoded]), kidney_scaler, k_features)
        label, prob = get_probability_label(kidney_model, X, "Chronic Kidney Disease", threshold)
        st.success(f"Prediction: {label} (Prob: {prob:.3f}, Threshold: {threshold})")

# ------------------------------
# Liver Disease Input
# ------------------------------
elif disease == "Liver Disease" and liver_model and liver_scaler:
    st.subheader("Liver Disease Input")
    input_data = {f: st.number_input(f, value=0.0) for f in l_features}
    if st.button("Predict Liver"):
        X = apply_scaler_and_order(pd.DataFrame([input_data]), liver_scaler, l_features)
        label = get_prediction_label(liver_model, X, "Liver Disease")
        st.success(f"Prediction: {label}")

































