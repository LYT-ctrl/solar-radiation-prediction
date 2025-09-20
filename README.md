# Solar Radiation Prediction — Machine Learning + Airflow Pipeline

## Objective
Predict hourly solar radiation/production from PV site data and ERA5 weather features.
This repo shows: (1) ML experimentation and model selection, (2) orchestration with Airflow (Astro), (3) reproducible project structure (Docker, requirements, tests).

## Pipeline (high level)
```mermaid
flowchart TD
    A[Collect PV + Weather] --> B[Clean & Preprocess]
    B --> C[Feature Engineering]
    C --> D[Train ML models]
    D --> E[Evaluate]
    E --> F[Persist model]
    F --> G[Orchestrate with Airflow]
Project structure
.
├── dags/
│   ├── solar_radiation_prediction.py   # Main Airflow DAG
│   └── exampledag.py                   # Example (can be removed)
├── include/
│   └── ml_pipeline.py                  # ETL/ML/eval/save functions
├── tests/                              # Pytest tests
├── requirements.txt                    # Python deps
├── packages.txt                        # OS-level deps (optional)
├── Dockerfile                          # Airflow + deps image
├── airflow_settings.yaml               # Airflow variables/connections
└── README.md
Data & preprocessing
PV production (kWh) and site metadata (lat/lon).
ERA5 via Open-Meteo API: temperature, humidity, pressure, wind, cloudcover, sunrise/sunset.
Merge by site and timestamp; handle missing values; time features (year, month, day, hour).
Modeling & evaluation
Models: LinearRegression, BayesianRidge, DecisionTree, RandomForest, SVR (and optional MLPRegressor).
Metrics: R2, MAE, MSE, RMSE.
Notes: scale where needed with pipelines; prefer time-series CV for temporal data.
Example results (from a representative run)
Model	MSE	RMSE	MAE	R²
Linear Regression	580	24.08	18.37	0.149
Decision Tree	607	24.65	17.62	0.109
Random Forest	596	24.43	17.56	0.125
SVR	675	25.98	15.84	0.009
Bayesian Ridge	580	24.08	18.37	0.149
MLP Regressor	554	23.54	17.78	0.187
Interpretation: non-linear models slightly outperform linear ones (R² ~0.18–0.19). Best next steps: TimeSeriesSplit, trig features (sin/cos hour, dayofyear), lags/rolling.
Airflow orchestration
DAG solar_radiation_prediction.py defines:
load_and_clean_data → 2) feature_selection → 3) train_model → 4) evaluate_model → 5) save_model
Run locally (Astro)
astro dev start
# Airflow UI: http://localhost:8080
# Enable the DAG: solar_radiation_prediction
Roadmap
TimeSeriesSplit CV; hyperparameter tuning (RF/SVR/MLP).
Advanced features: trig seasonality, lags/rolling, irradiance proxies.
XGBoost/LightGBM; model registry; Streamlit dashboard; cloud deployment.
Author
Yacine Tigrine — M2 AI & Engineering
GitHub: https://github.com/LYT-ctrl
README

3. Appuie sur **Entrée** après le dernier `__README__`.  
4. Vérifie le fichier :  
   ```bash
   cat README.md
EOF

EOF

EOF

EOF

EOF

