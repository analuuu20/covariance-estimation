# DSAP 2025 FINAL PROJECT: Model Comparison for Large-Scale Covariance Estimation for Financial Assets

This repository contains the implementation of a data science project focused on 
estimating high-dimensional covariance matrices using the Dynamic Conditional 
Correlation - Generalized Autoregressive Conditional Heteroskedasticity(DCC-GARCH) model as the baseline approach.

Ledoit–Wolf Shrinkage, Optimally-Shrunk Covariance Estimation(OAS), Principal Component Analysis (PCA) and Graphical Lasso (GLASSO) models will be executed to attempt other methodologies to estimate the variance of the sample. 

The covariance matrices obtained will be compared one to another by making out-of-sample testing with rolling windows to determine the error between the model estimation and the historical outcome to define the best-performing methodology compared to the baseline model.

The project includes:

- Data downloading and preprocessing
- DCC-GARCH estimation
- Ledoit–Wolf Shrinkage, Optimally-Shrunk Covariance Estimation(OAS), Principal Component Analysis (PCA) and Graphical Lasso (GLASSO) model implementation
- Out-of-sample performance comparison with alternative estimators
- Documentation, tests, and reproducible environment

# Project Structure
covariance-estimation/
├── README.md                 # comprehensive project documentation
├── PROPOSAL.md               # project proposal
├── requirements.txt          # python dependencies
├── setup.py                  # installation script
├── src/                      # source code
│ ├── init.py                 # 
│ ├── data_download.py        # data download from yfinance, cleaning and saving into a csv file
│ └── data_utils.py           # load csv file, returns computation and covariance models implementation
├── tests/                    # test suite
│ └── test_*.py
├── data/                     # sample data (csv files generated)
├── docs/
├── examples/                 # usage examples
└── results/


# Clone repository
  git clone https://github.com/analuuu20/covariance-estimation.git
  cd covariance-estimation

#  Installation requirements
pip install -r requirements.txt


# Usage

Examples will be provided in the `examples/` folder.



# Author
Ana Lucia Clavijo Martinez (AnaLucia.ClavijoMartinez@unil.ch)
Student number: 25411315





