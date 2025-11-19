"""
Utility functions for loading stock price time series data and computing its returns.

This module of the project aims to:
- Load CVS files and validate them
- Functions to convert the raw price data provided into log-returns or percentage returns to make information more manageable
- Merge both parts developped before into a convenience wrapper to load a CSV of stock prices and prepares its returns in one step
"""

#import relevant libraries
import pandas as pd
import numpy as np
from pathlib import Path

#csv loading function
def load_csv(path: str | Path, parse_dates: bool = True, index_col: int | None = None) -> pd.DataFrame:
    """
    Function to load a CSV file and convert it into a pandas DataFrame

    Args:
        path (str | Path):
            Path to the CSV file.
        parse_dates (bool, optional):
            Defined as True to automatically parse date columns into datetime objects.
            
        index_col (int | None, optional):
            Signals which column to use as DataFrame index. Usually 0 for time series.
            Default is None.

    Returns:
        a pd.DataFrame from the loaded CSV data.

    Raises:
        FileNotFoundError: If the file does not exist.

    Notes:
        - This function helps to have a consistent CSV loading process throughout the project.
        - Path objects are used to allow OS-independent file access and ensure reproducibility.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path, parse_dates=parse_dates, index_col=index_col)


def prepare_returns(
    df: pd.DataFrame,
    log: bool = True,
    dropna: bool = True
) -> pd.DataFrame:
    """
    Function to convert a DataFrame of price series into returns.

    Args:
        df (pd.DataFrame):
            DataFrame containing a price series, should have a datetime index.
        log (bool, optional):
            If True, then compute the log-returns: log(P_t) - log(P_{t-1}).
            If False, then compute percentage returns via pct_change().
            Default value is True.
        dropna (bool, optional):
            Defined as True to drop the NaN at the beginning produced by the differentiation.
            

    Returns:
        pd.DataFrame: DataFrame of returns with the same columns as `df`.

    Notes:
        - Log-returns are used due to its additive properties and better adjustment to a normal distribution.
    """
    if log:
        rets = np.log(df).diff()
    else:
        rets = df.pct_change()

    if dropna:
        rets = rets.dropna()

    return rets


def load_and_prepare(
    path: str | Path,
    price_cols: list[str] | None = None,
    log: bool = True
) -> pd.DataFrame:
    """
    Convenience wrapper: combines loading a CSV file and convert its price series into log-returns.

    Args:
        path (str | Path):
            Path to the CSV file containing the price series.
        price_cols (list[str] | None, optional):
            List of columns to keep to label (stock tickers).
            If None, the entire DataFrame is used.
            Default is None.
        log (bool, optional):
            Whether to compute log (True) or percentage returns (False). 
            Default is True.

    Returns:
        pd.DataFrame: the same DataFrame but with returns.

    Workflow:
        1. Load the CSV and use a time index (index_col=0).
        2. Indicate whether to filter the price columns to work with only the selected tickers.
        3. Compute returns using the function developped before `prepare_returns()`.

    Example of usage:
        load_and_prepare("prices.csv", log= True, price_cols=["AAPL", "GOOG"])
    """
    df = load_csv(path, index_col=0)

    if price_cols is not None:
        df = df[price_cols]

    return prepare_returns(df, log=log)
