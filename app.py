
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    PROJECT_DIR, "model_artifacts",
    "network_intrusion_model.joblib"
)

CONFIG_PATH = os.path.join(
    PROJECT_DIR, "model_artifacts",
    "model_config.json"
)

# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_model():
    loaded_model = joblib.load(MODEL_PATH)

    with open(CONFIG_PATH, "r") as file:
        loaded_config = json.load(file)

    return loaded_model, loaded_config

try:
    model, config = load_model()
    EXPECTED_FEATURES = config["input_features"]
    THRESHOLD = float(config["classification_threshold"])

    preprocessor = model.named_steps["preprocessor"]

    categorical_pipeline = (
        preprocessor.named_transformers_["categorical"]
    )

    encoder = categorical_pipeline.named_steps["encoder"]

    CATEGORICAL_OPTIONS = {
        feature: list(categories)
        for feature, categories in zip(
            ["proto", "service", "state"],
            encoder.categories_
        )
    }

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()

NUMERICAL_FEATURES = [
    feature for feature in EXPECTED_FEATURES
    if feature not in ["proto", "service", "state"]
]

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🛡️ Network Intrusion Detection System")

st.markdown(
    "Machine Learning-Based Network Traffic Classification"
)

st.caption(
    "UNSW-NB15 Dataset | Random Forest | Binary Classification"
)

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Single Prediction",
        "Bulk Prediction",
        "Model Performance",
        "Feature Importance"
    ]
)

st.sidebar.divider()

st.sidebar.success("Model Status: Loaded")
st.sidebar.write(f"**Model:** {config['model_name']}")
st.sidebar.write(f"**Input Features:** {len(EXPECTED_FEATURES)}")
st.sidebar.write(f"**Threshold:** {THRESHOLD}")

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.header("System Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Classification Model", "Random Forest")
    col2.metric("Input Features", len(EXPECTED_FEATURES))
    col3.metric("Detection Threshold", f"{THRESHOLD:.2f}")

    st.subheader("System Overview")

    st.markdown("""
    This application analyzes structured network traffic
    records and classifies them into two categories:

    - **Normal:** Legitimate network traffic.
    - **Malicious:** Traffic classified as potentially malicious.

    ### Available Modules

    1. Single Prediction — Analyze an individual traffic record.
    2. Bulk Prediction — Upload a CSV containing multiple records.
    3. Model Performance — Review model evaluation metrics.
    4. Feature Importance — Explore influential input features.
    """)

    st.info(
        "This application performs dataset-driven traffic "
        "classification. It does not capture live network packets "
        "or automatically block connections."
    )

# --------------------------------------------------
# SINGLE TRAFFIC PREDICTION
# --------------------------------------------------

elif page == "Single Prediction":

    st.header("Single Traffic Prediction")

    st.write(
        "Enter the network traffic attributes below to "
        "classify a traffic record."
    )

    st.info(
        "All 42 original model input features are required. "
        "Values are passed through the trained preprocessing pipeline."
    )

    with st.form("single_prediction_form"):

        st.subheader("1. Numerical Traffic Features")

        st.caption(
            "Enter the numerical values for the network traffic record."
        )

        numerical_values = {}

        with st.expander("Numerical Features (39)", expanded=True):

            cols = st.columns(2)

            for index, feature in enumerate(NUMERICAL_FEATURES):

                with cols[index % 2]:

                    numerical_values[feature] = st.number_input(
                        feature,
                        value=0.0,
                        step=0.01,
                        format="%.6f",
                        key=f"single_{feature}"
                    )

        st.subheader("2. Categorical Traffic Features")

        col1, col2, col3 = st.columns(3)

        with col1:
            proto = st.selectbox(
                "Protocol (proto)",
                CATEGORICAL_OPTIONS["proto"],
                index=CATEGORICAL_OPTIONS["proto"].index("tcp")
                if "tcp" in CATEGORICAL_OPTIONS["proto"] else 0
            )

        with col2:
            service = st.selectbox(
                "Service",
                CATEGORICAL_OPTIONS["service"],
                index=0
            )

        with col3:
            state = st.selectbox(
                "Connection State",
                CATEGORICAL_OPTIONS["state"],
                index=0
            )

        submitted = st.form_submit_button(
            "Analyze Traffic",
            use_container_width=True
        )

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    if submitted:

        try:
            input_data = numerical_values.copy()

            input_data["proto"] = proto
            input_data["service"] = service
            input_data["state"] = state

            input_df = pd.DataFrame(
                [input_data],
                columns=EXPECTED_FEATURES
            )

            probabilities = model.predict_proba(input_df)[0]

            malicious_index = list(model.classes_).index(1)

            malicious_probability = float(
                probabilities[malicious_index]
            )

            prediction = int(
                malicious_probability >= THRESHOLD
            )

            st.divider()
            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:

                if prediction == 1:
                    st.error("MALICIOUS TRAFFIC DETECTED")
                else:
                    st.success("NORMAL TRAFFIC DETECTED")

            with col2:
                st.metric(
                    "Malicious Probability",
                    f"{malicious_probability * 100:.2f}%"
                )

            st.progress(malicious_probability)

            st.caption(
                f"Classification threshold: {THRESHOLD:.2f}"
            )

            st.caption(
                "This is a model prediction, not a definitive "
                "security determination."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

# --------------------------------------------------
# PLACEHOLDER MODULES
# --------------------------------------------------

elif page == "Bulk Prediction":
    st.header("Bulk Traffic Prediction")
    st.info("This module will be implemented next.")

elif page == "Model Performance":
    st.header("Model Performance")
    st.info("This module will be implemented next.")

elif page == "Feature Importance":
    st.header("Feature Importance")
    st.info("This module will be implemented next.")
