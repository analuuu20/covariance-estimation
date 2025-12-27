# DSAP 2025 FINAL PROJECT: Model Comparison for Large-Scale Covariance Estimation for Financial Assets

This repository contains the implementation of a data science project focused on comparing different methodologies for
estimating high-dimensional covariance matrices. Starting with conventiona covariance amtrix as the baseline, the Dynamic Conditional Correlation - Generalized Autoregressive Conditional Heteroskedasticity (DCC-GARCH), Ledoit–Wolf Shrinkage and Graphical Lasso (GLASSO) models are used to estimate the variance of the sample.

The covariance matrices obtained will be compared one to another by static metrics and making out-of-sample testing with rolling windows to determine the error between the model estimation and the historical outcome to define the best-performing methodology compared to the baseline model.

The project includes:

- Data downloading and preprocessing
- Data splitting 
- Log-returns calculation 
- DCC-GARCH estimation
- Ledoit–Wolf Shrinkage model implementation
- Graphical Lasso (GLASSO) model implementation
- Static metrics for validation
- Rolling windows for dynamic validation 
- Modular design, csv files and plots for exposing results and a reproducible environment

# Research Question
  Which estimation model performs best for large-scale covariance matrices: DCC-GARCH, Ledoit-Wolf Shrinkage or Graphical Lasso?

## Setup

# Git repository
  git clone https://github.com/analuuu20/covariance-estimation.git
  cd covariance-estimation

# Virtual environment 
  conda env create -f environment.yml
  conda activate covariance-estimation

# Installation requirements
  Alternatively, for remote usage the libraries used are in the requirements.txt file.
  Use: 
  pip install -r requirements.txt

  For the "multigarch" library, installation is done through Github:
  pip install git+https://github.com/mechanicpanic/multigarch.git
   
  "multigarch" library created by Anna Smirnova. 

# Usage
  The project follows a src/-based package layout. Individual modules are not imported at package level to avoid unintended side effects and to ensure reproducibility across environments. All execution is orchestrated explicitly through:

  python main.py

  Expected output: Comparison between three models through static, dynamic metrics and a normalized score.

# Project Structure

| covariance_estimation/
|-- data/                     # sample data (csv files generated)
|-- results/                  # Output plots, matrices and metrics
|   |-- training              
|   |-- validation
|-- src/                      # source code
|   |-- covariance_estimation/
|   |   |-- __init__.py
|   |   |-- 01_data_download.py                    # Data loading
|   |   |-- 02_data_split.py                       # Data preprocessing
|   |   |-- 03_returns_calculation.py              # Log-returns calculation
|   |   |-- 04_baseline_matrix.py                  # Baseline computation
|   |   |-- 05_graphical_lasso_training.py         # Model training
|   |   |-- 06_ledoit_wolf_training.py             # Model training
|   |   |-- 07_dcc_garch_multigarch_training.py    # Model training
|   |   |-- 08_baseline_validation.py              # Baseline validation computation
|   |   |-- 09_full_validation_consolidated.py     # Evaluation metrics
|-- environment.yml           # dependencies
|-- PROPOSAL.md               # project proposal
|-- README.md                 # setup and usage
|-- main.py                   # main entry point
|-- requirements.txt          # python dependencies
|-- setup.py                  # installation script

# Results
Final consolidated ranking:
- Graphical Lasso: 0.000017 
- Ledoit-Wolf Shrinkage: 0.55
- DCC-GARCH: 0.85
- Winner: Ledoit-Wolf Shrinkage


# Author
Ana Lucia Clavijo Martinez (AnaLucia.ClavijoMartinez@unil.ch)
Student number: 25411315





