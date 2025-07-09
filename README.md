Thanks for the updated structure screenshots! Based on your full setup and explanation, here’s a complete and well-organized README.md file tailored for your 5G energy prediction dissertation project.

⸻


# 📡 5G Energy Consumption Prediction using Deep Learning

This repository presents the codebase for a dissertation project aimed at **predicting hourly energy consumption of 5G base stations** using deep learning. The models developed here are designed to help telecom operators optimize power usage and enhance sustainability across diverse network configurations.

---

## 🎯 Research Aim and Objectives

**Aim:**  
To develop a machine learning-based model that reliably forecasts the energy consumption (in kWh) of 5G base stations, enabling better energy-saving strategies in real-world telecom environments.

**Objectives:**
- Review ML/DL approaches used in 5G energy modeling.
- Acquire and process a comprehensive dataset (base station configs, load, usage).
- Build lag-based temporal features to capture consumption patterns.
- Design hybrid deep learning models (CNN + RNN/LSTM/GRU).
- Use GroupKFold cross-validation to evaluate generalization to unseen base stations.
- Analyze feature importance and conduct hypothesis testing on performance.
- Deliver actionable insights for network operators.

---

## 🧠 Problem Summary

Predicting 5G energy use is **not trivial**:
- ⚠️ **Temporal dependency**: Current usage depends on traffic/config in past hours.
- ⚠️ **Data leakage**: Base station records must not overlap between training/validation.
- ⚠️ **Overfitting risk**: Models may memorize instead of generalize.
- ⚠️ **Model selection**: Different models suit different data types (time vs static).
- ⚠️ **Tuning complexity**: Learning rates, batch size, dropout, layers — all matter!

---

## 🧪 Methodology & Design

### ✅ Feature Engineering:
- Uses **lagged features** (e.g., `Energy_T-1`, `load_diff2`) to model time patterns.
- Applies smoothing using **Savitzky-Golay** and **Second-Order Section (SOS)** filters.

### ✅ Model Architecture:
- Hybrid **CNN → LSTM / GRU / RNN** pipelines:
  - CNN learns local short-term shifts.
  - RNN-based layers model temporal dynamics.

### ✅ Data Leakage Prevention:
- **GroupKFold** split by base station ID prevents same-station info leak.

### ✅ Hyperparameter Tuning:
- Uses **Optuna** with:
  - `TPE` (Tree-structured Parzen Estimator) for Bayesian optimization.

---

## 🗂️ Folder Structure Overview

📁 5g-project-data/         # Raw base station, config, energy files (CSV)
📁 data/                    # Intermediate parquet files for modeling
└── preprocessed_data/ # Train/test splits
📁 artifacts/               # Trained model files (GRU, LSTM, RNN)
📁 logs/                    # Training logs
📁 results/                 # Per-model MAE scores
📁 Training-specs/          # Training durations and resource logs
📁 feature-imp/             # SHAP feature importance visualizations
📁 hypothesis_tests/        # One-sided t-test results (.json)
📁 comparison-graphs/       # MAE comparisons for visualization

🧠 main.py                # Entry point: runs full pipeline
📄 data_ingestion.py        # Converts CSVs to parquet
📄 data_preprocessing.py    # Cleans, smooths, adds lags
📄 training_prediction.py   # Model training & validation (GroupKFold)
📄 evaluation_comparison.py # Compare models vs baseline
📄 feature_importance_shap.py # SHAP-based model explanation
📄 hypothesis_testing.py    # Statistical testing (H0: MAE ≥ 1.5)
📄 settings.py              # Global constants and paths
📄 variables.yaml           # Experiment hyperparameters/configs
📄 Models.py                # Model definitions (MLP, CNN, LSTM, GRU)
📄 utils.py                 # Helpers

---

## ▶️ How to Run the Code

### 1️⃣ Install requirements
```bash
pip install -r requirements.txt

2️⃣ Launch full pipeline

python main.py

Everything from data ingestion, preprocessing, training, evaluation, model saving, and SHAP plots will run automatically.

⸻

📊 Key Outputs

Folder	Description
results/	MAE values for each model (GRU.csv, LSTM.csv…)
feature-imp/	SHAP value plots (*_shap_summary.png)
artifacts/	Final trained models
Training-specs/	Per-model training duration in JSON
hypothesis_tests/	One-tailed t-test for MAE < 1.5
comparison-graphs/	MAE comparison visualizations


⸻

📦 Dependencies

matplotlib==3.6.3  
numpy==1.23.5  
pandas==1.5.3  
PyYAML==6.0.2  
scikit_learn==1.7.0  
scipy==1.9.3  
seaborn==0.13.2  
shap==0.48.0  
tensorflow==2.18.0  
torch==2.6.0  
tqdm==4.64.1


⸻

📬 Contact - adahm7114@gmail.com

If you have questions or would like to contribute to the project, feel free to raise an issue or fork the repository.

⸻

Note: This repository was created as part of a master’s dissertation on predictive energy optimization in 5G networks.

---

