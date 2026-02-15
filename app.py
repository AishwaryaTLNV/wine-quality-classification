import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef
)

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Wine Quality Classification",
    layout="wide"
)

st.title("🍷 Wine Quality Classification App")
st.write(
    """
    This application demonstrates **multi-class classification**
    on the **UCI Wine Quality dataset** using multiple machine learning models.
    """
)

# ---------------------------------------------------------
# Load models and preprocessing objects safely
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    models = {}

    # Load sklearn models (safe everywhere)
    models["Logistic Regression"] = joblib.load("model/logistic.pkl")
    models["Decision Tree"] = joblib.load("model/decision_tree.pkl")
    models["kNN"] = joblib.load("model/knn.pkl")
    models["Naive Bayes"] = joblib.load("model/naive_bayes.pkl")
    models["Random Forest"] = joblib.load("model/random_forest.pkl")

    # Try loading XGBoost (may fail locally on macOS)
    try:
        models["XGBoost"] = joblib.load("model/xgboost.pkl")
    except Exception:
        st.warning(
            "⚠️ XGBoost could not be loaded locally (OpenMP missing). "
            "It will work correctly on Streamlit Community Cloud."
        )

    scaler = joblib.load("model/scaler.pkl")
    label_encoder = joblib.load("model/label_encoder.pkl")

    return models, scaler, label_encoder


models, scaler, label_encoder = load_artifacts()

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
st.sidebar.header("Model Selection")

model_name = st.sidebar.selectbox(
    "Choose a classification model",
    list(models.keys())
)

st.sidebar.markdown("---")
st.sidebar.write(
    "Upload a CSV file containing **test data**.\n\n"
    "- With `quality` column → metrics + confusion matrix\n"
    "- Without `quality` column → predictions only"
)

# ---------------------------------------------------------
# Dataset upload
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

# ---------------------------------------------------------
# Main logic
# ---------------------------------------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(data.head())

    # Separate features and labels if available
    if "quality" in data.columns:
        X = data.drop("quality", axis=1)
        y_true_original = data["quality"]
        y_true = label_encoder.transform(y_true_original)
        labels_available = True
    else:
        X = data
        labels_available = False

    model = models[model_name]

    # Apply scaling where required
    if model_name in ["Logistic Regression", "kNN", "Naive Bayes"]:
        X_input = scaler.transform(X)
    else:
        X_input = X

    # -----------------------------------------------------
    # Predictions
    # -----------------------------------------------------
    y_pred_encoded = model.predict(X_input)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    st.subheader("🔮 Predicted Wine Quality (First 10 Rows)")
    st.write(pd.Series(y_pred[:10], name="Predicted Quality"))

    # -----------------------------------------------------
    # Evaluation Metrics (only if true labels exist)
    # -----------------------------------------------------
    if labels_available:
        st.subheader("📊 Evaluation Metrics")

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred_encoded)
        precision = precision_score(
            y_true, y_pred_encoded, average="weighted", zero_division=0
        )
        recall = recall_score(y_true, y_pred_encoded, average="weighted")
        f1 = f1_score(y_true, y_pred_encoded, average="weighted")
        mcc = matthews_corrcoef(y_true, y_pred_encoded)

        # -------- SAFE AUC computation (edge-case handled) --------
        try:
            y_prob = model.predict_proba(X_input)
            if len(np.unique(y_true)) == y_prob.shape[1]:
                auc = roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class="ovr",
                    average="weighted"
                )
            else:
                auc = None
        except Exception:
            auc = None

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", round(accuracy, 3))
        col2.metric("AUC", round(auc, 3) if auc is not None else "N/A")
        col3.metric("MCC", round(mcc, 3))

        col4, col5, col6 = st.columns(3)
        col4.metric("Precision", round(precision, 3))
        col5.metric("Recall", round(recall, 3))
        col6.metric("F1 Score", round(f1, 3))

        # -------------------------------------------------
        # Confusion Matrix
        # -------------------------------------------------
        st.subheader("📉 Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred_encoded)

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        st.pyplot(fig)

    else:
        st.info(
            "ℹ️ No `quality` column found in the uploaded file. "
            "Metrics and confusion matrix are shown only when true labels are available."
        )

else:
    st.warning("Please upload a CSV file to proceed.")
