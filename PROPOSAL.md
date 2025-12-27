# Project Proposal: Model Comparison for Large-Scale Covariance Estimation for Financial Assets 

# Category:
Business & Finance Tools 

# Problem statement or motivation: 

Understanding portfolio performance, risk, and asset correlations can be challenging for an independent investor when dealing with multiple assets. Thankfully, alongside the  financial markets development, so the statistical models to predict its performance as well. Given the wide variety of models that exists now, this project aims to compare a range of models used for the variance estimation in order to find which one improve robustness of the estimation while reducing estimation error when working with high-dimensionality matrices.
        
# Planned approach and technologies:

The project will be following the class requirements, implemented in Python 3.10+ and using modular design principles. In that regard, the project will be developed according to the following structure:

    - Data collection: the primary data source will be historical daily adjusted stock prices extracted from Yahoo Finance. A sample composed by the S&P 500 index company stocks for the last five years will be used to test the different covariance estimators.

    - Data processing: the packages to be used are Pandas and request libraries to clean the prices time series and extract the data set in a csv file.

    - Covariance estimation methodologies: once the database is standardized, the covariance matrix will be computed using different methodologies. A conventional covariance matriz will be used as the baseline methodology and Ledoit–Wolf Shrinkage, Dynamic Conditional Correlation - Generalized Autoregressive Conditional Heteroskedasticity (DCC-GARCH) model and Graphical Lasso (GLASSO) models will be the other methodologies executed to attempt to outperform the variance estimation of the baseline. 

    - Result comparison: the covariance matrices obtained will be compared one to another by using static metrics and making out-of-sample testing with rolling windows to determine the error between the model estimation and the historical outcome to define the best-performing methodology.

# Expected challenges:
    
- Handling large-dimensional matrices efficiently and ensuring positive definiteness to guarantee the usability of the resulting matrix.

- Tuning regularization parameters to avoid overfitting or excessive noise.
    
# Success criteria: 

The implementation of the three proposed estimators for the covariances. Additionally, it is expected to find the best-performing estimator compared to the baseline method by out-of-sample prediction accuracy and a normalized score based on static metrics to compare each model results.

# Stretch goals: 

The main stretch goal would be to implement other more complex methodologies for covariance matrix computations to expand the analysis. Another minor stretch goal could be to work on an ensemble-methodology that combines the four methodologies used to see if the ensembles performs better than each methodology on its own.    
 
