import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    PROJECT_DIR, "model_artifacts", "network_intrusion_model.joblib"
)
CONFIG_PATH = os.path.join(
    PROJECT_DIR, "model_artifacts", "model_config.json"
)
SAMPLE_PATH = os.path.join(PROJECT_DIR, "sample_records.csv")

CATEGORICAL_FEATURES = ["proto", "service", "state"]


# --------------------------------------------------
# MODEL AND DATA LOADING
# --------------------------------------------------

@st.cache_resource
def load_model():
    """Load the fitted preprocessing + classification pipeline."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Model configuration not found: {CONFIG_PATH}")

    loaded_model = joblib.load(MODEL_PATH)
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        loaded_config = json.load(file)

    return loaded_model, loaded_config


@st.cache_data
def load_sample_data():
    """Load the small demonstration CSV."""
    if not os.path.exists(SAMPLE_PATH):
        raise FileNotFoundError("sample_records.csv was not found.")
    return pd.read_csv(SAMPLE_PATH)


try:
    model, config = load_model()

    EXPECTED_FEATURES = config["input_features"]
    THRESHOLD = float(config["classification_threshold"])

    if not isinstance(EXPECTED_FEATURES, list) or len(EXPECTED_FEATURES) != 42:
        raise ValueError("The model configuration must contain exactly 42 input features.")

    if not 0 <= THRESHOLD <= 1:
        raise ValueError("The classification threshold must be between 0 and 1.")

    if not hasattr(model, "predict_proba"):
        raise TypeError("The loaded model does not support probability predictions.")

    model_classes = list(model.classes_)
    if 0 not in model_classes or 1 not in model_classes:
        raise ValueError("The model must contain class labels 0 (Normal) and 1 (Malicious).")

    malicious_index = model_classes.index(1)

    preprocessor = model.named_steps["preprocessor"]
    categorical_pipeline = preprocessor.named_transformers_["categorical"]
    encoder = categorical_pipeline.named_steps["encoder"]

    CATEGORICAL_OPTIONS = {
        feature: [str(value) for value in categories]
        for feature, categories in zip(CATEGORICAL_FEATURES, encoder.categories_)
    }

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


NUMERICAL_FEATURES = [
    feature for feature in EXPECTED_FEATURES
    if feature not in CATEGORICAL_FEATURES
]


def predict_traffic(input_df):
    """Return binary predictions and malicious-class probabilities."""
    probabilities = model.predict_proba(input_df)
    malicious_probabilities = probabilities[:, malicious_index]
    predictions = (malicious_probabilities >= THRESHOLD).astype(int)
    return predictions, malicious_probabilities


def get_single_select_options(feature):
    """Include an unseen sample value so it can still be displayed and tested."""
    options = list(CATEGORICAL_OPTIONS[feature])
    current_value = st.session_state.get(f"single_{feature}")
    if current_value is not None and str(current_value) not in options:
        options.append(str(current_value))
    return options


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🛡️ Network Intrusion Detection System")
st.markdown("Machine Learning-Based Network Traffic Classification")
st.caption("UNSW-NB15 Dataset | Random Forest | Binary Classification")
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
        "Feature Importance",
    ],
)

st.sidebar.divider()
st.sidebar.success("Model Status: Loaded")
st.sidebar.write(f"**Model:** {config.get('model_name', 'Random Forest')}")
st.sidebar.write(f"**Input Features:** {len(EXPECTED_FEATURES)}")
st.sidebar.write(f"**Threshold:** {THRESHOLD:.2f}")


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
    st.markdown(
        """
        This application analyzes structured network traffic records
        and classifies them into two categories:

        - **Normal:** Legitimate network traffic.
        - **Malicious:** Traffic classified as potentially malicious.

        ### Available Modules

        1. Single Prediction — Analyze an individual traffic record.
        2. Bulk Prediction — Upload a CSV containing multiple records.
        3. Model Performance — Review the official test-set evaluation.
        4. Feature Importance — Explore influential input features.
        """
    )

    st.info(
        "This application performs dataset-driven traffic classification. "
        "It does not capture live network packets or automatically block connections."
    )


# --------------------------------------------------
# SINGLE TRAFFIC PREDICTION
# --------------------------------------------------

elif page == "Single Prediction":
    st.header("Single Traffic Prediction")
    st.write(
        "Enter the network traffic attributes below to classify a traffic record."
    )

    st.subheader("Load a Sample Traffic Record")

    try:
        sample_data = load_sample_data()

        required_sample_columns = set(EXPECTED_FEATURES + ["label"])
        missing_sample_columns = required_sample_columns - set(sample_data.columns)

        if missing_sample_columns:
            st.warning(
                "The sample CSV is missing required columns: "
                + ", ".join(sorted(missing_sample_columns))
            )
        else:
            sample_choices = {}

            for label, name in [(0, "Normal"), (1, "Malicious")]:
                matching_records = sample_data[sample_data["label"] == label]

                if not matching_records.empty:
                    record = matching_records.iloc[0]
                    record_id = record["id"] if "id" in record.index else "N/A"
                    choice = f"{name} Traffic (Record ID: {record_id})"
                    sample_choices[choice] = record

            if sample_choices:
                selected_sample = st.selectbox(
                    "Choose a sample record",
                    list(sample_choices.keys()),
                    key="sample_record_choice",
                )

                if st.button("Load Sample Record", use_container_width=True):
                    sample = sample_choices[selected_sample]

                    for feature in NUMERICAL_FEATURES:
                        value = pd.to_numeric(sample[feature], errors="coerce")
                        st.session_state[f"single_{feature}"] = (
                            0.0 if pd.isna(value) else float(value)
                        )

                    for feature in CATEGORICAL_FEATURES:
                        value = sample[feature]
                        st.session_state[f"single_{feature}"] = (
                            "" if pd.isna(value) else str(value)
                        )

                    st.session_state["sample_loaded"] = selected_sample
                    st.rerun()
            else:
                st.warning("No Normal or Malicious sample records were found.")

    except FileNotFoundError:
        st.warning(
            "sample_records.csv was not found. You can still enter values manually."
        )
    except Exception as error:
        st.error(f"Sample loading failed: {error}")

    if st.session_state.get("sample_loaded"):
        st.success(f"Loaded: {st.session_state['sample_loaded']}")

    st.divider()
    st.info(
        "All 42 original model input features are required. "
        "You can enter them manually or load a sample record above."
    )

    with st.form("single_prediction_form"):
        st.subheader("1. Numerical Traffic Features")
        st.caption("Enter the 39 numerical network traffic attributes.")

        numerical_values = {}

        with st.expander("Numerical Features (39)", expanded=True):
            cols = st.columns(2)

            for index, feature in enumerate(NUMERICAL_FEATURES):
                with cols[index % 2]:
                    numerical_values[feature] = st.number_input(
                        feature,
                        step=0.01,
                        format="%.6f",
                        key=f"single_{feature}",
                    )

        st.subheader("2. Categorical Traffic Features")
        col1, col2, col3 = st.columns(3)

        with col1:
            proto = st.selectbox(
                "Protocol (proto)",
                get_single_select_options("proto"),
                key="single_proto",
            )

        with col2:
            service = st.selectbox(
                "Service",
                get_single_select_options("service"),
                key="single_service",
            )

        with col3:
            state = st.selectbox(
                "Connection State",
                get_single_select_options("state"),
                key="single_state",
            )

        submitted = st.form_submit_button(
            "Analyze Traffic",
            use_container_width=True,
        )

    if submitted:
        try:
            input_data = numerical_values.copy()
            input_data.update(
                {
                    "proto": proto,
                    "service": service,
                    "state": state,
                }
            )

            input_df = pd.DataFrame([input_data], columns=EXPECTED_FEATURES)
            predictions, malicious_probabilities = predict_traffic(input_df)

            prediction = int(predictions[0])
            malicious_probability = float(malicious_probabilities[0])

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
                    f"{malicious_probability * 100:.2f}%",
                )

            st.progress(malicious_probability)
            st.caption(f"Classification threshold: {THRESHOLD:.2f}")
            st.caption(
                "This is a model prediction, not a definitive security determination."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# BULK TRAFFIC PREDICTION
# --------------------------------------------------

elif page == "Bulk Prediction":
    st.header("Bulk Traffic Prediction")
    st.write(
        "Upload a CSV containing multiple network traffic records. "
        "The model will classify each record using the same preprocessing "
        "pipeline and threshold as Single Prediction."
    )

    st.info(
        "The CSV must contain all 42 model input features. Additional columns "
        "(such as id, attack_cat or label) are allowed and are retained in the "
        "download, but are not used as model inputs."
    )

    uploaded_file = st.file_uploader(
        "Upload Network Traffic CSV",
        type=["csv"],
        key="bulk_csv",
    )

    if uploaded_file is not None:
        try:
            bulk_data = pd.read_csv(uploaded_file)

            if bulk_data.empty:
                st.warning("The uploaded CSV contains no records.")
            elif not bulk_data.columns.is_unique:
                st.error("The uploaded CSV contains duplicate column names.")
            else:
                missing_features = [
                    feature for feature in EXPECTED_FEATURES
                    if feature not in bulk_data.columns
                ]

                if missing_features:
                    st.error(
                        "The CSV is missing these required model features: "
                        + ", ".join(missing_features)
                    )
                else:
                    input_df = bulk_data[EXPECTED_FEATURES].copy()

                    # Convert numerical inputs safely. Invalid or blank
                    # numerical cells become NaN and are handled by the
                    # fitted pipeline's median imputer.
                    for feature in NUMERICAL_FEATURES:
                        input_df[feature] = pd.to_numeric(
                            input_df[feature], errors="coerce"
                        )

                    # Reject records with no supplied model input at all.
                    empty_rows = input_df.isna().all(axis=1)
                    if empty_rows.any():
                        row_numbers = (np.flatnonzero(empty_rows.to_numpy()) + 2).tolist()
                        st.error(
                            f"{int(empty_rows.sum())} record(s) contain no usable "
                            "model features. CSV row number(s): "
                            + ", ".join(map(str, row_numbers[:20]))
                        )
                    else:
                        predictions, malicious_probabilities = predict_traffic(
                            input_df
                        )

                        results = bulk_data.copy()
                        results["Malicious_Probability"] = (
                            malicious_probabilities * 100
                        ).round(2)
                        results["Prediction"] = np.where(
                            predictions == 1, "Malicious", "Normal"
                        )

                        st.success(
                            f"Successfully analyzed {len(results):,} network traffic records."
                        )

                        st.subheader("Prediction Summary")
                        col1, col2, col3 = st.columns(3)

                        col1.metric("Total Records", f"{len(results):,}")
                        col2.metric(
                            "Normal Traffic",
                            f"{int((predictions == 0).sum()):,}",
                        )
                        col3.metric(
                            "Malicious Traffic",
                            f"{int((predictions == 1).sum()):,}",
                        )

                        st.subheader("Prediction Results")
                        st.dataframe(
                            results,
                            use_container_width=True,
                            hide_index=True,
                        )

                        csv_output = results.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download Prediction Results",
                            data=csv_output,
                            file_name="network_predictions.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                        st.caption(
                            f"Predictions use a malicious-probability threshold of "
                            f"{THRESHOLD:.2f}. The model does not use any label or "
                            "attack-category columns supplied in the uploaded file."
                        )

        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty or is not a valid CSV.")
        except Exception as error:
            st.error(f"Bulk prediction failed: {error}")


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":
    st.header("Model Performance")
    st.write(
        "The following results are from the model's evaluation on the "
        "official UNSW-NB15 test set. These are fixed evaluation results, "
        "not metrics recalculated from uploaded prediction files."
    )

    # Verified official test-set results from the project evaluation.
    accuracy = 0.8710
    precision = 0.8178
    recall = 0.9852
    f1_score = 0.8937

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
    col2.metric("Precision", f"{precision * 100:.2f}%")
    col3.metric("Recall", f"{recall * 100:.2f}%")
    col4.metric("F1-Score", f"{f1_score * 100:.2f}%")

    st.caption(
        "Evaluation set: 82,332 records | Normal: 37,000 | Malicious: 45,332 "
        "| Classification threshold: 0.50"
    )

    st.divider()
    st.subheader("Confusion Matrix")

    # Rows are actual classes; columns are predicted classes.
    confusion_matrix = pd.DataFrame(
        [[27053, 9947], [673, 44659]],
        index=["Actual Normal", "Actual Malicious"],
        columns=["Predicted Normal", "Predicted Malicious"],
    )

    st.dataframe(
        confusion_matrix,
        use_container_width=True,
    )

    matrix_col1, matrix_col2 = st.columns(2)
    matrix_col1.metric("True Negatives (TN)", "27,053")
    matrix_col1.metric("False Positives (FP)", "9,947")
    matrix_col2.metric("False Negatives (FN)", "673")
    matrix_col2.metric("True Positives (TP)", "44,659")

    st.markdown(
        """
        **How to read the confusion matrix**

        - **True Negative:** Normal traffic correctly classified as Normal.
        - **False Positive:** Normal traffic incorrectly flagged as Malicious.
        - **False Negative:** Malicious traffic incorrectly classified as Normal.
        - **True Positive:** Malicious traffic correctly detected.
        """
    )

    st.divider()
    st.subheader("Classification Report")

    classification_report = pd.DataFrame(
        {
            "Precision": [0.98, 0.82, 0.90, 0.89],
            "Recall": [0.73, 0.99, 0.86, 0.87],
            "F1-Score": [0.84, 0.89, 0.86, 0.87],
            "Support": [37000, 45332, 82332, 82332],
        },
        index=["Normal", "Malicious", "Macro Average", "Weighted Average"],
    )

    st.dataframe(
        classification_report.style.format(
            {
                "Precision": "{:.2f}",
                "Recall": "{:.2f}",
                "F1-Score": "{:.2f}",
                "Support": "{:,.0f}",
            }
        ),
        use_container_width=True,
    )

    st.warning(
        "The model detects most malicious records (98.52% malicious recall), "
        "but it also flags some normal records as malicious. The 9,947 false "
        "positives are an important limitation to consider."
    )


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

elif page == "Feature Importance":
    st.header("Feature Importance")
    st.write(
        "This module presents permutation feature importance calculated on "
        "the validation set using three repeats and F1-score."
    )

    st.info(
        "Permutation importance measures how much the validation F1-score "
        "decreases when a feature is shuffled. A larger value indicates a "
        "greater contribution to the model's validation performance."
    )

    # Verified permutation importance results from the project evaluation.
    importance_data = [
        ("ct_srv_dst", 0.00418, 0.00024),
        ("ct_dst_src_ltm", 0.00370, 0.00011),
        ("service", 0.00347, 0.00021),
        ("smean", 0.00314, 0.00027),
        ("sbytes", 0.00257, 0.00024),
        ("ct_srv_src", 0.00255, 0.00005),
        ("dmean", 0.00209, 0.00031),
        ("synack", 0.00184, 0.00027),
        ("dbytes", 0.00163, 0.00021),
        ("ct_dst_ltm", 0.00133, 0.00017),
        ("dload", 0.00121, 0.00029),
        ("ct_src_ltm", 0.00107, 0.00024),
        ("ct_dst_sport_ltm", 0.00106, 0.00023),
        ("dttl", 0.00106, 0.00026),
        ("dloss", 0.00103, 0.00009),
    ]

    importance_df = pd.DataFrame(
        importance_data,
        columns=["Feature", "Mean Importance", "Standard Deviation"],
    )

    st.subheader("Top 15 Features")
    chart_data = importance_df.set_index("Feature")[["Mean Importance"]]
    st.bar_chart(chart_data, horizontal=True)

    st.subheader("Importance Scores")
    st.dataframe(
        importance_df.style.format(
            {
                "Mean Importance": "{:.5f}",
                "Standard Deviation": "{:.5f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "These are validation-set permutation importance scores, not "
        "probabilities or percentages. Correlated features can share predictive "
        "information, so importance rankings should be interpreted accordingly."
    )
