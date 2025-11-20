# Project Proposal: Model Comparison for Large-Scale Covariance Estimation for Financial Assets 

# Category:
Business & Finance Tools 

# Problem statement or motivation: 

Understanding portfolio performance, risk, and asset correlations can be challenging for an independent investor when dealing with multiple assets. Thankfully, alongside the  financial markets development, so the statistical models to predict its performance as well. Given the wide variety of models that exists now, this project aims to compare a range of models used for the variance estimation in order to find which one improve robustness of the estimation while reducing estimation error when working with high-dimensionality matrices.
        
# Planned approach and technologies:

The project will be following the subject requirements, implemented in Python 3.10+ and using modular design principles. In that regard, the project will be developed according to the following structure:

    - Data collection: the primary data source will be historical daily adjusted stock prices extracted from Yahoo Finance. A sample composed by S&P 500 company stocks for the last five years will be used to test the different covariance estimators.

    - Data processing: the packages to be used are Pandas and request libraries to clean the prices time series and extract the data set in a csv file.

    - Covariance estimation methodologies: once the database is standardized, volatility and correlations will be computed using different methodologies. A DCC-GARCH model will be used as the baseline methodology and a Ledoit–Wolf Shrinkage, Optimally-Shrunk Covariance Estimation (OAS), Principal Component Analysis (PCA) and Graphical Lasso (GLASSO) models will be executed to attempt other methodologies to estimate the variance of the sample. 

    - Result comparison: the covariance matrices obtained will be compared one to another by making out-of-sample testing with rolling windows to determine the error between the model estimation and the historical outcome to define the best-performing methodology.

# Expected challenges:
    
- Handling large-dimensional matrices efficiently and ensuring positive definiteness to guarantee the usability of the resulting matrix.

- Tuning regularization parameters to avoid overfitting or excessive noise.
    
# Success criteria: 

The implementation of at least three estimators for the covariances. Additionally, it is expected to find the best-performing estimator compared to the baseline method by out-of-sample prediction accuracy for portfolio variance.

# Stretch goals: 

The main stretch goal would be to implement other more complex methodologies for covariance matrix computations to expand the analysis. Another minor stretch goal could be to work on an ensemble-methodology that combines the four methodologies used to see if the ensembles performs better than each methodology on its own.    
 
