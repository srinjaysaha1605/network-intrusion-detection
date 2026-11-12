# 🛡️ Network Intrusion Detection System

### Machine Learning-Based Network Traffic Classification

A machine learning-based network intrusion detection system that classifies network traffic as **Normal** or **Malicious** using the UNSW-NB15 dataset and a trained Random Forest classifier.

<p align="center">
  <a href="https://network-intrusion-detection-ds.streamlit.app/?utm_source=chatgpt.com#network-intrusion-detection-system">
    <strong>🚀 Live Application</strong>
  </a>
  &nbsp; • &nbsp;
  <a href="https://research.unsw.edu.au/projects/unsw-nb15-dataset">
    <strong>📊 Dataset Source</strong>
  </a>
</p>

---

## 📌 Overview

Network intrusion detection plays an important role in identifying potentially malicious activities within network traffic. This project implements a supervised machine learning approach to distinguish legitimate network connections from potentially malicious ones.

The system uses the **UNSW-NB15 dataset** to train and evaluate multiple classification algorithms, ultimately selecting a Random Forest classifier based on its validation performance.

The trained model is deployed through an interactive Streamlit application that supports individual traffic classification, bulk CSV prediction, model performance analysis, and feature importance visualization.

### Key Features

- **Single Prediction:** Classify individual network traffic records.
- **Bulk Prediction:** Upload CSV files containing multiple network traffic records and generate predictions.
- **Malicious Traffic Detection:** Estimate the probability of a traffic record being malicious.
- **Model Performance:** Explore the model's official test-set evaluation metrics and confusion matrix.
- **Feature Importance:** Examine the features contributing to predictive performance using permutation importance.
- **CSV Export:** Download bulk prediction results for further analysis.

---

## 🧰 Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model Serialization | Joblib |
| Data Visualization | Matplotlib, Seaborn |
| Web Application | Streamlit |
| Dataset | UNSW-NB15 |
| Deployment | Streamlit Community Cloud |

---

## 📊 Dataset

This project uses the **UNSW-NB15 dataset**, developed by the Australian Centre for Cyber Security (ACCS).

The dataset contains a combination of normal network traffic and simulated attack activities, with network flow characteristics suitable for intrusion detection research.

**Official dataset:**  
https://research.unsw.edu.au/projects/unsw-nb15-dataset

### Dataset Information

| Description | Details |
|---|---:|
| Training records | 175,341 |
| Testing records | 82,332 |
| Total input features | 42 |
| Numerical features | 39 |
| Categorical features | 3 |
| Target variable | `label` |
| Classification type | Binary |

The target variable is mapped as follows:

| Label | Classification |
|---|---|
| 0 | Normal |
| 1 | Malicious |

The `id` and `attack_cat` columns are excluded from model inputs. The `label` column is used as the target during training and evaluation, not as an input feature.

---

## ⚙️ Machine Learning Workflow

The project follows a structured machine learning pipeline.

### 1. Data Preprocessing

- Separated input features and target labels.
- Removed identifier and target-derived columns.
- Identified numerical and categorical features.
- Applied median imputation and standard scaling to numerical features.
- Applied most-frequent imputation and One-Hot Encoding to categorical features.
- Used `handle_unknown="ignore"` to handle previously unseen categorical values.
- Created a stratified 80:20 training-validation split.

### 2. Model Development

Three supervised classification algorithms were implemented and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

### 3. Cross-Validation

Five-fold cross-validation was performed to evaluate model consistency across different data splits.

### 4. Hyperparameter Tuning

GridSearchCV with stratified three-fold cross-validation and F1-score optimization was used to tune the classification models.

### 5. Model Selection

Random Forest was selected based on its validation performance.

### 6. Threshold Analysis

Different classification thresholds were examined to understand the trade-off between precision and recall.

The final operating threshold was retained at **0.50**.

### 7. Model Export

The complete fitted preprocessing and classification pipeline was serialized using Joblib to preserve the same transformations during deployment.

---

## 📈 Model Performance

The selected Random Forest model was evaluated on the official UNSW-NB15 test set.

<details>
<summary><strong>Official Test Results</strong></summary>

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 87.10% |
| Precision | 81.78% |
| Recall | 98.52% |
| F1-Score | 89.37% |

**Evaluation details:**

- Test records: 82,332
- Normal records: 37,000
- Malicious records: 45,332
- Classification threshold: 0.50

</details>

### Confusion Matrix

| | Predicted Normal | Predicted Malicious |
|---|---:|---:|
| Actual Normal | 27,053 | 9,947 |
| Actual Malicious | 673 | 44,659 |

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Normal | 0.98 | 0.73 | 0.84 |
| Malicious | 0.82 | 0.99 | 0.89 |

**Interpretation:**

The model achieves a malicious traffic recall of 98.52%, indicating that it detects most malicious records in the official test set.

However, 9,947 normal records were incorrectly classified as malicious. These false positives highlight an important limitation and indicate scope for improving the model's ability to distinguish legitimate traffic from malicious activity.

*Note: These are fixed results from the official test-set evaluation. They are not recalculated using files uploaded through the deployed application.*

---

## 🔍 Feature Importance

Permutation feature importance was calculated using the validation set, with three repeats and F1-score as the evaluation metric.

It measures the decrease in validation F1-score when an individual feature is randomly shuffled.

### Top 15 Features

| Rank | Feature | Mean Importance |
|---:|---|---:|
| 1 | ct_srv_dst | 0.00418 |
| 2 | ct_dst_src_ltm | 0.00370 |
| 3 | service | 0.00347 |
| 4 | smean | 0.00314 |
| 5 | sbytes | 0.00257 |
| 6 | ct_srv_src | 0.00255 |
| 7 | dmean | 0.00209 |
| 8 | synack | 0.00184 |
| 9 | dbytes | 0.00163 |
| 10 | ct_dst_ltm | 0.00133 |
| 11 | dload | 0.00121 |
| 12 | ct_src_ltm | 0.00107 |
| 13 | ct_dst_sport_ltm | 0.00106 |
| 14 | dttl | 0.00106 |
| 15 | dloss | 0.00103 |

Higher importance values indicate a greater contribution to validation performance when compared with shuffled feature values.

These scores are not probabilities or percentages. Correlated features may share predictive information, so the rankings should be interpreted accordingly.

---

## 🖥️ Streamlit Application

The trained model is integrated into an interactive Streamlit dashboard.

**Live application:**  
https://network-intrusion-detection-ds.streamlit.app/

### Application Modules

| Module | Description |
|---|---|
| Dashboard | Provides an overview of the intrusion detection system. |
| Single Prediction | Classifies an individual network traffic record. |
| Bulk Prediction | Processes multiple records from an uploaded CSV file. |
| Model Performance | Displays official test-set evaluation metrics and classification results. |
| Feature Importance | Visualizes validation-set permutation feature importance. |

### Bulk Prediction Output

The bulk prediction module generates:

- Predicted classification for each record.
- Malicious traffic probability.
- Original input columns alongside prediction results.
- Downloadable CSV containing the predictions.

---

## 📁 Project Structure

```text
network-intrusion-detection/
│
├── app.py
├── requirements.txt
├── sample_records.csv
│
├── model_artifacts/
│   ├── network_intrusion_model.joblib
│   └── model_config.json
│
└── assets/
```

### Important Files

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application. |
| `requirements.txt` | Python dependencies. |
| `sample_records.csv` | Sample network traffic records for testing. |
| `network_intrusion_model.joblib` | Serialized preprocessing and Random Forest pipeline. |
| `model_config.json` | Model configuration, input features, and classification threshold. |
| `assets/` | Application assets. |

---

## 🚀 Running Locally

### Prerequisites

- Python 3.11
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/srinjaysaha1605/network-intrusion-detection.git
```

### 2. Navigate to the Project Directory

```bash
cd network-intrusion-detection
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

**Note:** Keep the `model_artifacts/` directory and its contents in their original locations. The application requires the exported model and configuration files to load successfully.

---

## 📂 Testing Bulk Predictions

A sample file named `sample_records.csv` is included in the repository.

To test bulk prediction:

1. Launch the Streamlit application.
2. Navigate to the Bulk Prediction module.
3. Upload `sample_records.csv`.
4. Run the prediction.
5. Review the predicted classes and malicious probabilities.
6. Download the generated prediction CSV.

Uploaded files should contain the 42 expected model input features. Additional metadata columns such as `id`, `attack_cat`, and `label` can be included; they are not used as model inputs.

The `label` column, when present, is useful for comparing predictions with known ground truth, but it is not required for prediction.

---

## ⚠️ Limitations

- The system performs binary classification and does not identify individual attack categories.
- The model is trained on the UNSW-NB15 dataset; performance on real-world network traffic may differ.
- False positives remain a significant limitation, with 9,947 normal records incorrectly classified as malicious in the official test evaluation.
- Changes in network traffic characteristics and distribution shift may affect prediction performance.
- Predictions should be treated as decision-support information rather than definitive evidence of an intrusion.
- The application is an academic machine learning project, not a production-grade network security monitoring solution.

---

## 🔮 Future Improvements

Potential improvements include:

- Reducing false positive rates through further model optimization.
- Evaluating additional ensemble and boosting algorithms.
- Exploring feature selection and dimensionality reduction.
- Investigating data drift and model generalization.
- Integrating real-time network traffic monitoring.
- Exploring explainable AI techniques for individual predictions.

---

## 🎓 Project Information

**Project Title:** Network Intrusion Detection System Using Machine Learning  
**Course:** Master of Computer Applications (MCA)  
**Project Type:** Final-Year Project-Based Learning (PBL)  
**Dataset:** UNSW-NB15  
**Selected Algorithm:** Random Forest Classifier  
**Application Framework:** Streamlit

---

## 👨‍💻 Author

**Srinjay Saha**

MCA Student | AI/ML & Full-Stack Developer

- **GitHub:** https://github.com/srinjaysaha1605

---

## 📜 Acknowledgements

The project uses the UNSW-NB15 dataset developed by the Australian Centre for Cyber Security (ACCS) for network intrusion detection research.

Dataset and research information:  
https://research.unsw.edu.au/projects/unsw-nb15-dataset

---

<p align="center">
  <strong>Built with Python, Machine Learning & Streamlit.</strong>
  <br>
  <sub>Network Intrusion Detection System · MCA Final-Year PBL Project</sub>
</p>
