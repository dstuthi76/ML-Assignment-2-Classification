import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(
    page_title="ML Classification Model Comparison",
    page_icon="📊",
    layout="wide"
)

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "kNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest": "model/random_forest.joblib"
}

FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]

def load_model(name):
    return joblib.load(MODEL_FILES[name])

st.title("📊 Breast Cancer Classification – Model Comparison")
st.caption("BITS Pilani M.Tech AIML/DSE | Machine Learning Assignment – 2")

st.markdown("""
### Objective
Compare five classification algorithms on the UCI Breast Cancer Wisconsin
(Diagnostic) dataset and evaluate them using Accuracy, AUC, Precision,
Recall, F1 Score and Matthews Correlation Coefficient (MCC).
""")

uploaded = st.file_uploader(
    "Upload test data CSV",
    type=["csv"],
    help="Upload the supplied test_data.csv or another test CSV with the same 30 feature columns and a diagnosis column."
)

if uploaded is None:
    st.info("Upload test_data.csv to display predictions and evaluation metrics.")
    st.stop()

df = pd.read_csv(uploaded)
df.columns = [c.strip() for c in df.columns]

# Accept either UCI/sklearn feature names or the standard WDBC names.
if "diagnosis" not in df.columns:
    st.error("The CSV must contain a 'diagnosis' column containing M/B labels.")
    st.stop()

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error(f"Missing required feature columns: {missing}")
    st.stop()

X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
if X.isna().any().any():
    st.error("Some feature values are missing or non-numeric.")
    st.stop()

y_text = df["diagnosis"].astype(str).str.strip().str.upper()
mapping = {"M": 1, "B": 0, "MALIGNANT": 1, "BENIGN": 0}
y = y_text.map(mapping)

if y.isna().any():
    st.error("Diagnosis must use M/B (or Malignant/Benign) labels.")
    st.stop()

y = y.astype(int)

selected = st.selectbox("Select classification model", list(MODEL_FILES.keys()))
model = load_model(selected)

pred = model.predict(X)
proba = model.predict_proba(X)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y, pred),
    "AUC": roc_auc_score(y, proba),
    "Precision": precision_score(y, pred, zero_division=0),
    "Recall": recall_score(y, pred, zero_division=0),
    "F1 Score": f1_score(y, pred, zero_division=0),
    "MCC": matthews_corrcoef(y, pred)
}

st.subheader(f"Evaluation Metrics – {selected}")
cols = st.columns(6)
for col, (metric, value) in zip(cols, metrics.items()):
    col.metric(metric, f"{value:.4f}")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y, pred, labels=[0, 1])
cm_df = pd.DataFrame(
    cm,
    index=["Actual Benign", "Actual Malignant"],
    columns=["Predicted Benign", "Predicted Malignant"]
)
st.dataframe(cm_df, use_container_width=True)

st.subheader("Classification Report")
report = classification_report(
    y, pred, target_names=["Benign", "Malignant"],
    output_dict=True, zero_division=0
)
st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

out = df.copy()
out["Predicted Diagnosis"] = np.where(pred == 1, "M", "B")
out["Malignant Probability"] = np.round(proba, 4)

st.subheader("Prediction Results")
st.dataframe(out, use_container_width=True)

st.download_button(
    "Download predictions as CSV",
    out.to_csv(index=False).encode("utf-8"),
    "predictions.csv",
    "text/csv"
)
