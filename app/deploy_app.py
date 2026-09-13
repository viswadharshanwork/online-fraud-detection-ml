import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Online Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "final_random_forest.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "fraud_threshold.pkl"

# ============================================================
# FEATURE ORDER
# ============================================================

FEATURES = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
    "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]

# ============================================================
# LOAD MODEL AND THRESHOLD
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_threshold():
    return float(joblib.load(THRESHOLD_PATH))


model = load_model()
threshold = load_threshold()

# ============================================================
# HEADER
# ============================================================

st.title("💳 Online Fraud Detection System")

st.markdown("### Machine Learning Based Transaction Risk Analysis")

st.write(
    "This system uses a Random Forest machine learning model "
    "to identify potentially fraudulent credit card transactions."
)

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🤖 Model Information")

    st.write("**Algorithm:** Random Forest")
    st.write("**Trees:** 200")
    st.write("**Class Weight:** Balanced")

    st.divider()

    st.write("### 📊 Final Test Performance")

    st.metric("ROC-AUC", "94.30%")
    st.metric("PR-AUC", "81.89%")
    st.metric("Fraud Precision", "94.44%")
    st.metric("Fraud Recall", "71.58%")

    st.divider()

    st.caption(
        "This application is an ML-based fraud screening "
        "prototype and should not be used as the sole basis "
        "for financial decisions."
    )

# ============================================================
# TRANSACTION DETAILS
# ============================================================

st.subheader("💰 Transaction Details")

col1, col2 = st.columns(2)

with col1:

    time = st.number_input(
        "Time (seconds)",
        min_value=0.0,
        value=0.0,
        format="%.2f"
    )

with col2:

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0,
        format="%.2f"
    )

# ============================================================
# PCA FEATURES
# ============================================================

st.subheader("🔢 Transaction Features")

st.info(
    "V1–V28 are anonymized PCA-transformed features "
    "provided by the original dataset."
)

v_values = {}

feature_columns = st.columns(4)

for i in range(1, 29):

    with feature_columns[(i - 1) % 4]:

        v_values[f"V{i}"] = st.number_input(
            f"V{i}",
            value=0.0,
            format="%.6f"
        )

st.divider()

# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔍 CHECK TRANSACTION",
    use_container_width=True,
    type="primary"
):

    input_data = {
        "Time": time
    }

    for i in range(1, 29):
        input_data[f"V{i}"] = v_values[f"V{i}"]

    input_data["Amount"] = amount

    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURES
    )

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    probability = float(
        model.predict_proba(input_df)[0][1]
    )

    prediction = int(
        probability >= threshold
    )

    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    st.subheader("🎯 Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:

        st.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )

    with result_col2:

        st.metric(
            "Decision Threshold",
            f"{threshold * 100:.2f}%"
        )

    with result_col3:

        if prediction == 1:

            st.metric(
                "Model Decision",
                "⚠️ FRAUD"
            )

        else:

            st.metric(
                "Model Decision",
                "✅ LEGITIMATE"
            )

    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    if probability >= 0.80:

        risk_level = "🔴 HIGH RISK"

    elif probability >= threshold:

        risk_level = "🟠 MEDIUM RISK"

    else:

        risk_level = "🟢 LOW RISK"

    st.markdown(f"## {risk_level}")

    # --------------------------------------------------------
    # DECISION MESSAGE
    # --------------------------------------------------------

    if prediction == 1:

        st.error("🚨 POTENTIAL FRAUD DETECTED")

        st.warning(
            "The transaction probability exceeds "
            "the model's decision threshold."
        )

    else:

        st.success("✅ TRANSACTION APPEARS LEGITIMATE")

        st.info(
            "The transaction probability is below "
            "the model's decision threshold."
        )

    # --------------------------------------------------------
    # PROBABILITY BAR
    # --------------------------------------------------------

    st.write("### Fraud Probability")

    st.progress(
        min(max(float(probability), 0.0), 1.0)
    )

# ============================================================
# MODEL EXPLAINABILITY
# ============================================================

st.divider()

st.subheader("🧠 Model Insights")

st.write(
    "The Random Forest model uses the transaction features "
    "to distinguish between legitimate and fraudulent "
    "transactions."
)

feature_importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

top_features = feature_importance.head(10)

st.write("### 🔝 Top 10 Important Features")

importance_df = (
    top_features
    .sort_values(ascending=True)
    .reset_index()
)

importance_df.columns = ["Feature", "Importance"]

st.bar_chart(
    importance_df,
    x="Feature",
    y="Importance",
    horizontal=True
)

st.write(
    "These values represent the global feature importance "
    "learned by the Random Forest model. Higher values "
    "indicate greater contribution to the model's overall "
    "decision-making."
)