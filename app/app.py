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
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"


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
# LOAD MODEL, THRESHOLD AND DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_threshold():
    return float(joblib.load(THRESHOLD_PATH))


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model = load_model()
threshold = load_threshold()
df = load_data()


# ============================================================
# SESSION STATE
# ============================================================

if "sample_transaction" not in st.session_state:
    st.session_state["sample_transaction"] = None

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
        "This application is an ML-based fraud screening "
        "prototype and should not be used as the sole basis "
        "for financial decisions."
    )


# ============================================================
# TRANSACTION INPUT
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
        "Use a real transaction from the dataset for demonstration."
    )

    sample_type = st.selectbox(
        "Choose transaction type:",
        ["Legitimate Transaction", "Fraudulent Transaction"]
    )

    if sample_type == "Legitimate Transaction":
        sample_rows = df[df["Class"] == 0]
        sample_key = "legitimate_sample_index"
    else:
        sample_rows = df[df["Class"] == 1]
        sample_key = "fraudulent_sample_index"

    if sample_key not in st.session_state:
        st.session_state[sample_key] = 0

    sample_index = st.number_input(
        "Sample transaction index",
        min_value=0,
        max_value=len(sample_rows) - 1,
        step=1,
        key=sample_key
    )

    if st.button(
        "📂 LOAD SAMPLE TRANSACTION",
        use_container_width=True
    ):

        selected_transaction = sample_rows.iloc[int(sample_index)]

        st.session_state["sample_transaction"] = (
            selected_transaction[FEATURES].to_dict()
        )

        st.session_state["actual_class"] = int(
            selected_transaction["Class"]
        )

        st.session_state["prediction_result"] = None

        st.success("Sample transaction loaded successfully!")


# ============================================================
# CURRENT SAMPLE
# ============================================================

sample_data = st.session_state["sample_transaction"]


# ============================================================
# TRANSACTION DETAILS
# ============================================================

st.subheader("💰 Transaction Details")

col1, col2 = st.columns(2)

if sample_data is not None:
    default_time = float(sample_data["Time"])
    default_amount = float(sample_data["Amount"])
else:
    default_time = 0.0
    default_amount = 100.0


with col1:

    time = st.number_input(
        "Time (seconds)",
        min_value=0.0,
        value=default_time,
        format="%.2f"
    )


with col2:

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=default_amount,
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

if sample_data is not None:

    for i in range(1, 29):
        v_values[f"V{i}"] = float(sample_data[f"V{i}"])

    st.success("Real dataset feature values loaded.")

else:

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

    # Save result so it remains available after Streamlit reruns.
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


    # --------------------------------------------------------
    # SAMPLE GROUND TRUTH
    # --------------------------------------------------------

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
