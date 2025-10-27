# Exoplanet Detection using machine learning

## Overview  

This project focuses on detecting exoplanets using data from NASA's Kepler mission. It involves:

- `1. Data Acquisition and Preprocessing`: Querying the NASA Exoplanet Archive for Kepler Objects of Interest (KOIs), confirmed planet names, and planetary system parameters. The data is merged, cleaned, and prepared for machine learning.
- `2. Exoplanet Classification`: Building and evaluating machine learning models to classify KOIs as confirmed exoplanets, false positives, or candidates. A stacking classifier is used for optimal performance, with predictions applied to unresolved candidates.

The dataset includes ~8,000 KOIs with features like orbital period, transit depth, stellar temperature, and more. The classifier achieves high accuracy (~0.93+ on test sets) in distinguishing confirmed planets from false positives.
This is a complete end-to-end pipeline for exoplanet detection, suitable for astronomy enthusiasts, students, or researchers exploring transit photometry data.
Features

- `Data Pipeline`: Automated querying and merging from NASA Exoplanet Archive using astroquery.
- `Exploratory Data Analysis (EDA)`: Visualizations of key features and class distributions.
- `Machine Learning Models`: Ensemble methods (Random Forest, XGBoost, CatBoost) stacked with Logistic Regression.
- `Model Evaluation`: Cross-validation, SHAP explanations, and hyperparameter tuning via Bayesian optimization.
- `Candidate Prediction`: Applies the trained model to unresolved candidates and outputs predictions with probabilities.

## Project Structure  

Exoplanet-Detection-using-machine-learning/  
├── README.md                 # This file  
├── data/                     # Output directory for datasets (created during runtime)  
│   ├── kepler_objects_of_interset(kois).csv  
│   ├── kepler_confirmed_planet_names.csv  
│   ├── planetory_systems.csv  
│   ├── final_df.csv  
├── download_koi_data.ipynb     # Data download, merging, and preprocessing  
├── exoplanet_classifier.ipynb  # EDA, modeling, and prediction  
├── fitted.joblib               # All trained models  
├── metrics.csv                 # Score and Optimization time of each model  
├── model.joblib                # Trained best performing model  
├── candidates_predictions.csv  # Generated predictions  
└── requirements.txt            # Dependencies  

## Installation  

### 1. Clone the Repositor:  

- git clone https://github.com/atharvaingale88/Exoplanet-Detection-using-machine-learning.git  
- cd Exoplanet-Detection-using-machine-learning  

### 2. Create a Virtual Environment (recommended):  

#### Using conda (as in the notebooks)  

- conda create -n exoplanet_detection_env  
- conda activate exoplanet_detection_env  

#### Or using venv  

- python -m venv exoplanet_detection_env  
- source exoplanet_detection_env/bin/activate  # On Windows: exoplanet_detection_env\Scripts\activate  

### 3. Install Dependencies:  

pip install -r requirements.txt  

## Usage  

### Run the notebooks in sequence using Jupyter:  

#### 1. Start Jupyter:  

jupyter notebook  

#### 2. Run `download_koi_data.ipynb`:  

- Queries NASA Exoplanet Archive (requires internet).  
- Outputs cleaned `final_df.csv` (~8K rows, 50 columns).  
- Filters to confirmed planets and false positives; handles candidates separately.  

#### 3. Run `exoplanet_classifier.ipynb`:  

- Loads `final_df.csv` and performs EDA (e.g., histograms, correlation plots).   
- Preprocesses data (imputation, scaling, encoding).  
- Trains and evaluates models; saves `fitted.joblib` abd `model.joblib`.  
- Predicts on candidates and saves `candidates_predictions.csv` with probabilities.  

## Results  

### Model Performance (Stacking Classifier)  

Accuracy: 0.9385898407884761  
F1-Score: 0.927484333034915  
ROC-AUC: 0.9765751518519046  

### Candidate Predictions  

- From 1,349 candidates: 759 predicted false positives (56.26%), 590 potential exoplanets (44.74%).  
- High-confidence predictions (>0.9 prob) can guide follow-up observations.  

## Limitations & Future Work  

- `Data Scope`: Limited to Kepler transit data; extend to TESS or radial velocity methods.  
- `Imbalanced Classes`: Handled via stratification, but synthetic oversampling (e.g., SMOTE) could improve.  
- `Hyperparameters`: Bayesian search used; grid search for deeper tuning.  
- `Deployment`: Wrap in a Streamlit app for interactive predictions.  

## Acknowledgments  

- NASA Exoplanet Archive for public data.  
- scikit-learn, XGBoost, LightGBM, CatBoost, and SHAP for ML tools.  
