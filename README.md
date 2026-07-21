# 🧠 EdTech & Digital Wellbeing: Mental Health Risk Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-red.svg)](https://streamlit.io/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NawafChukkan/Edtech-digital-wellbeing-ml/blob/main/notebooks/digital_wellbeing_analysis.ipynb)

An end-to-end Machine Learning solution designed to predict student psychological burnout ("Good", "Moderate", "At-risk") utilizing non-invasive digital telemetry and behavioral usage habits. Developed as part of the M.Sc. in Data Analytics curriculum (Domain Applications) at the **National College of Ireland**.

---

## 📌 Business & Societal Impact
Educational institutions face unprecedented student dropout rates tied to digital fatigue and psychological distress. This project develops an interpretable machine learning pipeline that identifies individuals approaching burnout thresholds, allowing EdTech platforms and universities to deploy proactive interventions (e.g., notification throttling, automated wellness prompts) before a student crisis occurs.

---

## 🛠️ Tech Stack & Methodology
*   **Language & Environment:** Python 3.10+, Google Colab, Jupyter Notebooks
*   **Data Engineering:** Pandas, NumPy (Imputation, One-Hot Encoding, Stratified Splitting)
*   **Machine Learning:** Scikit-Learn (Logistic Regression, Random Forest Classifier)
*   **Model Interpretability:** Feature Importance Analysis, Confusion Matrix Validation
*   **Deployment:** Streamlit Interactive Web Application

---

## 📊 Key Results & Findings

The project benchmarked a linear baseline against a non-linear ensemble model to predict the target `wellbeing_band`.

*   **Champion Model:** The Random Forest Classifier achieved **99.29% overall accuracy** and a **95% recall rate** on the critical "At-risk" target class.
*   **Behavioral Drivers:** Feature importance extraction proved that **Daily Screen Time**, **Anxiety Score**, and **Notification Frequency** are the dominant indicators of psychological fatigue, far outweighing the specific digital platform being used.

| Model | Accuracy | Macro F1-Score | At-Risk Recall |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 91.71% | 0.90 | 0.88 |
| **Random Forest (Champion)** | **99.29%** | **0.99** | **0.95** |

> **🔬 Data Science Insight:** The exceptionally high accuracy (99.29%) suggests the presence of target leakage within the dataset. It is highly probable that the original `wellbeing_band` classification was mathematically derived from features like the anxiety score, which the Random Forest successfully reverse-engineered. This highlights the importance of critical data provenance evaluation in real-world ML applications.

---

## 📁 Repository Structure

```text
Edtech-digital-wellbeing-ml/
│
├── README.md                                       <- Project documentation
├── requirements.txt                                <- Python dependencies
├── social_media_screentime_mental_health_2026.csv  <- Dataset for execution
│
├── notebooks/
│   └── digital_wellbeing_analysis.ipynb            <- Main Jupyter/Colab Notebook 
│
├── app/
│   └── app.py                                      <- Streamlit Web Dashboard
│
└── assets/
    ├── eda_overview.png                            <- Exploratory Data Analysis
    ├── confusion_matrix.png                        <- Model Evaluation Matrix
    └── feature_importance.png                      <- Top Predictive Features
