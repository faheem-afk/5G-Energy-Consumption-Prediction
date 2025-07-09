📡 5G Energy Consumption Prediction (Dissertation Project)

This repository contains the complete source code and resources for a Master’s dissertation project focused on forecasting energy consumption of 5G base stations using advanced deep learning models. The pipeline is fully automated via main.py and built to explore the impact of time-dependence, model generalization, and real-world deployment constraints.

⸻

🎯 Research Aim and Objectives

Aim:
To develop a reliable machine learning-based model capable of accurately predicting the energy consumption of 5G base stations, assisting network operators in enhancing their energy-saving strategies.

Objectives:
	•	Review literature on ML methods for 5G energy prediction.
	•	Select and implement a robust data science approach.
	•	Gather and preprocess a comprehensive dataset with traffic load, base station configs, and power usage.
	•	Design, train, and evaluate ML/DL models including CNN, LSTM, GRU.
	•	Validate model generalization on unseen base stations.
	•	Interpret results and offer practical recommendations.

⸻

🧠 Problem Summary
	•	Temporal Dependency: Energy usage depends not only on current inputs but also on previous load, power levels, and configurations. Models assuming i.i.d. inputs (like traditional ML) underperform.
	•	Data Leakage: Same base station data in both training and validation leads to false confidence. Solved via GroupKFold.
	•	Overfitting Risk: Models memorize station-specific trends. Generalization tested on unseen stations.
	•	Imbalanced Data: Some base stations are overrepresented.
	•	Architecture Selection: Compared MLP, CNN, LSTM, GRU, hybrid models.
	•	Hyperparameter Tuning: Used Optuna with TPE and CMA-ES samplers.

⸻

🏗️ Design Overview
	•	Feature Engineering: Lagged energy, rolling load, and elapsed time as time-series predictors. Smoothing filters: Savitzky-Golay, Second Order Sections.
	•	Model Architecture: Hybrid CNN-RNN (GRU, LSTM, RNN) structure to learn local and sequential patterns.
	•	Evaluation Protocol: GroupKFold split ensures no base station appears in both train/val.
	•	Hyperparameter Optimization: Optuna was used to automate tuning.
	•	Model Interpretability: SHAP used for post-hoc explanations.
	•	Statistical Testing: One-sided t-tests to evaluate model MAE < 1.5 kWh.

⸻

🏃 How to Run

# Install dependencies
pip install -r requirements.txt

# Run the entire pipeline
python main.py

📁 All logs, models, graphs, and results will be saved automatically to their respective folders.

⸻

📂 Folder Structure Overview

📁 5g-project-data/        # Raw base station, config, energy files (CSV)
📁 data/
 └── preprocessed_data/   # Train/test parquet splits
📁 artifacts/              # Trained model folders (GRU, LSTM, RNN)
📁 logs/                   # Training logs
📁 results/                # CSV MAE per model
📁 Training-specs/         # Avg epoch time, training device info
📁 hypothesis_tests/       # One-sided t-test results (.json)
📁 feature-imp/            # SHAP visualizations (summary plots)
📁 comparison-graphs/      # MAE comparisons (e.g. GRU vs LSTM)

📄 main.py                 # Entry point; runs entire pipeline
📄 data_ingestion.py       # Converts CSVs to parquet
📄 data_preprocessing.py   # Cleans, filters, adds lags
📄 training_prediction.py  # GroupKFold training + test evaluation
📄 evaluation_comparison.py# Compare models vs baseline
📄 feature_importance_shap.py # SHAP impact plots
📄 hypothesis_testing.py   # One-sided t-test (H0: MAE >= 1.5)
📄 Models.py               # Defines MLP, CNN, LSTM, GRU
📄 pipeline.py             # Pipeline connectivity
📄 settings.py             # Global constants and folder paths
📄 variables.yaml          # Hyperparameters and control switches
📄 utils.py                # Misc helpers (e.g., filters)


⸻

🔬 Core Functionalities
	•	Data Ingestion (data_ingestion.py): Reads raw CSV files from 5g-project-data/ and saves them as parquet in data/.
	•	Preprocessing (data_preprocessing.py): Cleans data, smoothens traffic with Savitzky-Golay & SOS, adds lagged variables.
	•	Model Training (training_prediction.py): Trains each model over GroupKFold, stores predictions, tracks validation & test MAE.
	•	Model Evaluation (evaluation_comparison.py): Compares models visually and numerically.
	•	Explainability (feature_importance_shap.py): SHAP plots for each model.
	•	Significance Test (hypothesis_testing.py): Performs 1-sided t-test (H0: MAE >= 1.5).

⸻

📦 Requirements

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

💬 Author & Credits

This project was developed as part of a Master’s dissertation on 5G energy consumption prediction. The project integrates deep learning, explainable AI, and statistical testing in a real-world forecasting problem.

For questions or contributions, feel free to open an issue or fork the repo.

⸻

📘 License

MIT License. See LICENSE file for details.
