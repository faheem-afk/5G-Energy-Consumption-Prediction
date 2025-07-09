Thanks for sharing the updated visual of your folder structure! Based on it, here’s the updated and complete README.md content with the correct folder and file explanations incorporated:

⸻

📡 5G Energy Consumption Prediction

This repository contains the complete implementation of a machine learning-based system for predicting hourly energy consumption of 5G base stations. It is designed to help telecom operators anticipate power demands and improve energy efficiency through intelligent, data-driven modeling.

⸻

🎯 Research Aim and Objectives

Aim:
To develop a reliable ML-based model capable of accurately predicting the energy consumption (in kWh) of 5G base stations. The goal is to support network operators in improving their energy-saving strategies.

Objectives:
	•	📚 Review ML methods applied to energy prediction in 5G networks.
	•	🧪 Select an appropriate Data Science methodology.
	•	📥 Gather and process datasets with features like configuration, traffic load, and energy usage.
	•	🧠 Design, train, and evaluate multiple ML models (MLP, CNN, LSTM, GRU).
	•	🌐 Test models on unseen base stations to ensure generalization.
	•	📊 Analyze results and recommend strategies based on performance.

⸻

❗ Problem Analysis Summary
	•	Time-Dependency: Energy usage depends on lagged features such as traffic, configuration, and past energy levels.
	•	Data Leakage Risk: If base stations appear in both train/validation sets, the model might memorize rather than generalize.
	•	Overfitting: Inflated performance on validation sets may not transfer to new stations.
	•	Imbalanced Data: Overrepresented base stations can bias the model.
	•	Model Selection: No universally optimal architecture; various deep models tested.
	•	Hyperparameter Tuning: Manual tuning is error-prone; automated strategies are used (Optuna, TPE, CMA-ES).

⸻

🧠 Design Highlights
	•	✅ Temporal features engineered: via lagging traffic, energy, and settings.
	•	✅ Hybrid Deep Models: CNN for short-term pattern extraction + LSTM/GRU for temporal learning.
	•	✅ GroupKFold Strategy: Avoids leakage by keeping base stations exclusive to each fold.
	•	✅ Smart Hyperparameter Optimization: Done using Optuna (TPE & CMA-ES samplers).

⸻

📂 Folder Structure Overview

📁 5g-project-data/       # Raw base station, config, energy files (CSV)
📁 data/
  └── preprocessed_data/ # Train/test splits as parquet
📁 artifacts/             # Trained model files (GRU, LSTM, RNN)
📁 logs/                  # Training logs
📁 results/               # Per-model MAE results
📁 Training-specs/        # Training durations and device usage logs
📁 feature-imp/           # SHAP feature impact visualizations
📁 hypothesis_tests/      # One-sided t-test results for model significance
📁 comparison-graphs/     # MAE comparison graphs


⸻

🧾 Script Overview

Script	Purpose
main.py	🚀 Entry point: Runs the full pipeline
data_ingestion.py	📥 Converts raw CSVs to parquet format
data_preprocessing.py	🧹 Cleans data, adds lags, applies smoothing filters
training_prediction.py	🧠 Trains models (GRU, LSTM, etc.) using GroupKFold
evaluation_comparison.py	📊 Compares trained models with the baseline
feature_importance_shap.py	🔍 Visualizes SHAP values for feature explanation
hypothesis_testing.py	📈 Runs 1-sample t-test (MAE < 1.5) and outputs significance
Models.py	🏗️ Contains definitions for MLP, CNN, LSTM, GRU
settings.py	⚙️ Global paths and constant configurations
variables.yaml	📑 Hyperparameter ranges and training variables
utils.py	🧰 Helper functions
logger.py	📝 Logging utility for experiment tracking
pipeline.py	🔄 Utility for chaining major steps together


⸻

🏁 How to Run
	1.	Clone the repository

git clone https://github.com/yourusername/5g-energy-prediction.git
cd 5g-energy-prediction


	2.	Install dependencies

pip install -r requirements.txt


	3.	Run the pipeline

python main.py


	4.	Outputs will be saved to:
	•	results/ – MAE scores per model
	•	artifacts/ – Trained model binaries
	•	feature-imp/ – SHAP plots
	•	hypothesis_tests/ – p-value analysis
	•	Training-specs/ – Time taken per fold/device info

⸻

🔬 Requirements

Listed in requirements.txt:

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

📈 Sample Output Visuals
	•	MAE Comparison between models
	•	SHAP plots showing feature impact
	•	One-sided t-test JSON verdicts
	•	Training time summary logs

⸻

📬 Contact

For queries or feedback, reach out via GitHub or email the project author.

⸻

Let me know if you want this README exported as a file or tailored for a specific GitHub theme (e.g. dark/light markdown, badges, license, contribution instructions, etc.).
