import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

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
# DEMO SAMPLES
# These are real rows from the Credit Card Fraud Detection dataset.
# They are stored here so the deployed app does not need the
# 143 MB CSV file.
# ============================================================

LEGITIMATE_SAMPLE = {
    "Time": 0.0,
    "V1": -1.3598071336738,
    "V2": -0.0727811733098497,
    "V3": 2.53634673796914,
    "V4": 1.37815522427443,
    "V5": -0.338320769942518,
    "V6": 0.462387777762292,
    "V7": 0.239598554061257,
    "V8": 0.0986979012610507,
    "V9": 0.363786969611213,
    "V10": 0.0907941719789316,
    "V11": -0.551599533260813,
    "V12": -0.617800855762348,
    "V13": -0.991389847235408,
    "V14": -0.311169353699879,
    "V15": 1.46817697209427,
    "V16": -0.470400525259478,
    "V17": 0.207971241929242,
    "V18": 0.0257905801985591,
    "V19": 0.403992960255733,
    "V20": 0.251412098239705,
    "V21": -0.018306777944153,
    "V22": 0.277837575558899,
    "V23": -0.110473910188767,
    "V24": 0.0669280749146731,
    "V25": 0.128539358273528,
    "V26": -0.189114843888824,
    "V27": 0.133558376740387,
    "V28": -0.0210530534538215,
    "Amount": 149.62
}

FRAUD_SAMPLE = {
    "Time": 406.0,
    "V1": -2.312226542,
    "V2": 1.951992,
    "V3": -1.6098507,
    "V4": 3.997906,
    "V5": -0.5221879,
    "V6": -1.42654532,
    "V7": -2.5373873,
    "V8": 1.39165725,
    "V9": -2.7700893,
    "V10": -2.7722721,
    "V11": 3.2020332,
    "V12": -2.8999074,
    "V13": -0.59522188,
    "V14": -4.289254,
    "V15": 0.389724120,
    "V16": -1.1407472,
    "V17": -2.8300557,
    "V18": -0.01682247,
    "V19": 0.4169557,
    "V20": 0.126910559,
    "V21": 0.5172324,
    "V22": -0.03504937,
    "V23": -0.4652111,
    "V24": 0.32019820,
    "V25": 0.04451917,
    "V26": 0.1778398,
    "V27": 0.26114500,
    "V28": -0.14327587,
    "Amount": 0.00
}

# ============================================================
# LOAD MODEL
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
# SESSION STATE
# ============================================================

if "sample_data" not in st.session_state:
    st.session_state["sample_data"] = None

if "actual_class" not in st.session_state:
    st.session_state["actual_class"] = None

if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None

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
        "This application is an ML-based fraud screening prototype "
        "and should not be used as the sole basis for financial decisions."
    )

# ============================================================
# TRANSACTION SOURCE
# ============================================================

st.subheader("📥 Transaction Input")

input_mode = st.radio(
    "Choose transaction source:",
    ["Manual Input", "Sample Transaction"],
    horizontal=True
)

# ============================================================
# SAMPLE TRANSACTION MODE
# ============================================================

if input_mode == "Sample Transaction":

    st.info(
        "Load a real labeled transaction from the Credit Card Fraud "
        "Detection dataset for demonstration."
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "✅ LOAD LEGITIMATE SAMPLE",
            use_container_width=True
        ):
            for feature_name, value in LEGITIMATE_SAMPLE.items():
                st.session_state[f"input_{feature_name}"] = float(value)
            st.session_state["sample_data"] = LEGITIMATE_SAMPLE.copy()
            st.session_state["actual_class"] = 0
            st.session_state["prediction_result"] = None

    with col2:
        if st.button(
            "🚨 LOAD FRAUD SAMPLE",
            use_container_width=True
        ):
            for feature_name, value in FRAUD_SAMPLE.items():
                st.session_state[f"input_{feature_name}"] = float(value)
            st.session_state["sample_data"] = FRAUD_SAMPLE.copy()
            st.session_state["actual_class"] = 1
            st.session_state["prediction_result"] = None

    if st.session_state["sample_data"] is not None:
        st.success("Sample transaction loaded. Click CHECK TRANSACTION.")

# ============================================================
# CURRENT SAMPLE
# ============================================================

sample_data = st.session_state["sample_data"]

# ============================================================
# TRANSACTION DETAILS
# ============================================================

st.subheader("💰 Transaction Details")

col1, col2 = st.columns(2)

default_time = (
    float(sample_data["Time"])
    if sample_data is not None
    else 0.0
)

default_amount = (
    float(sample_data["Amount"])
    if sample_data is not None
    else 100.0
)

if "input_Time" not in st.session_state:
    st.session_state["input_Time"] = default_time

if "input_Amount" not in st.session_state:
    st.session_state["input_Amount"] = default_amount

with col1:
    time = st.number_input(
        "Time (seconds)",
        min_value=0.0,
        format="%.2f",
        key="input_Time"
    )

with col2:
    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        format="%.2f",
        key="input_Amount"
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
    feature_name = f"V{i}"

    default_value = (
        float(sample_data[feature_name])
        if sample_data is not None
        else 0.0
    )

    if f"input_{feature_name}" not in st.session_state:
        st.session_state[f"input_{feature_name}"] = default_value

    with feature_columns[(i - 1) % 4]:
        v_values[feature_name] = st.number_input(
            feature_name,
            format="%.6f",
            key=f"input_{feature_name}"
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

    input_data = {"Time": time}

    for i in range(1, 29):
        input_data[f"V{i}"] = v_values[f"V{i}"]

    input_data["Amount"] = amount

    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURES
    )

    probability = float(
        model.predict_proba(input_df)[0][1]
    )

    prediction = int(probability >= threshold)

    st.session_state["prediction_result"] = {
        "probability": probability,
        "prediction": prediction
    }

# ============================================================
# DISPLAY PREDICTION RESULT
# ============================================================

result = st.session_state["prediction_result"]

if result is not None:

    probability = result["probability"]
    prediction = result["prediction"]

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
        st.metric(
            "Model Decision",
            "⚠️ FRAUD" if prediction == 1
            else "✅ LEGITIMATE"
        )

    # Risk level
    if probability >= 0.80:
        risk_level = "🔴 HIGH RISK"
    elif probability >= threshold:
        risk_level = "🟠 MEDIUM RISK"
    else:
        risk_level = "🟢 LOW RISK"

    st.markdown(f"## {risk_level}")

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

    st.write("### Fraud Probability")
    st.progress(
        min(max(float(probability), 0.0), 1.0)
    )

    # Show dataset ground truth only for the demo samples.
    if (
        input_mode == "Sample Transaction"
        and st.session_state["actual_class"] is not None
    ):
        actual = st.session_state["actual_class"]

        st.divider()
        st.write("### Dataset Ground Truth")

        if actual == 1:
            st.error("Actual label: FRAUD")
        else:
            st.success("Actual label: LEGITIMATE")

        if actual == prediction:
            st.success("✅ Model prediction matches the dataset label.")
        else:
            st.warning(
                "⚠️ Model prediction does not match this sample's label."
            )

# ============================================================
# MODEL EXPLAINABILITY
# ============================================================

st.divider()
st.subheader("🧠 Model Insights")

st.write(
    "The Random Forest model uses the transaction features "
    "to distinguish between legitimate and fraudulent transactions."
)

st.write("### 🔝 Top 10 Important Features")

importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values("Importance", ascending=False)
    .head(10)
    .sort_values("Importance")
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
ax.set_title("Top 10 Feature Importances")
st.pyplot(fig)

st.caption(
    "Higher values indicate greater contribution to the model's "
    "overall decision-making."
)
