"""
Utility functions to download, transform, clean, and validate financial time series data.

The data pipeline follows these steps:

1. Retrieve the list of S&P 500 constituents from Wikipedia.
2. Download historical OHLCV (Open, High, Low, Close, Adjusted Close, Volume)
   price data for each ticker using the yfinance API.
3. Convert the MultiIndex structure produced by yfinance into a flat, tabular dataset.
4. Clean the dataset by removing invalid values, duplicates, and non-numeric entries.
5. Validate the resulting dataset using basic quality checks.
6. Save the cleaned dataset into CSV format for further analysis.

This script is intended to provide a reproducible and academically rigorous data ingestion workflow
for covariance estimation or other financial econometrics applications.
"""

import pandas as pd
import yfinance as yf
import requests


# =====================================================================
# 1. Retrieve S&P 500 tickers from Wikipedia
# =====================================================================

def get_sp500_tickers():
    """
    Extract S&P 500 tickers from Wikipedia using a robust HTTP request and
    robust table selection (to avoid issues caused by structural changes
    in the Wikipedia HTML formatting).

    The function:
    - Sends an HTTP request with a custom User-Agent (avoids 403 Forbidden errors).
    - Parses all HTML tables found in the page.
    - Selects the table containing the 'Symbol' column, which identifies listed tickers.
    - Converts tickers to a Yahoo Finance–compatible format (replacing '.' with '-').

    Returns
    -------
    list[str]
        The list of ticker symbols of the S&P 500 constituents.
    """

    from io import StringIO
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    # Use a realistic user-agent string to prevent server rejection
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch S&P 500 list. Status code: {response.status_code}")

    # Wrap HTML content in a buffer to avoid FutureWarnings
    dfs = pd.read_html(StringIO(response.text))

    # Select the correct table by column inspection
    table = None
    for df in dfs:
        if "Symbol" in df.columns:
            table = df
            break

    if table is None:
        raise ValueError("Could not find S&P 500 table with 'Symbol' column.")

    tickers = table["Symbol"].tolist()
    tickers = [t.replace(".", "-") for t in tickers]  # Required for Yahoo Finance API

    return tickers



# =====================================================================
# 2. Download price data using yfinance
# =====================================================================

def download_prices(tickers, period="5y"):
    """
    Download historical OHLCV price data for the given list of tickers.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols.
    period : str
        Time span for which historical data should be downloaded.
        Follows the official yfinance API format (e.g., '1y', '5y', 'max').

    Returns
    -------
    DataFrame
        MultiIndex DataFrame with levels:
        - Date (index)
        - Variable name (Open, High, Low, Close, Adj Close, Volume)
        - Ticker symbol
    """
    print(f"Downloading data for {len(tickers)} tickers over period = '{period}'...")
    df = yf.download(tickers, period=period, group_by="ticker", auto_adjust=False)
    return df



# =====================================================================
# 3. Flatten MultiIndex DataFrame to a simple table
# =====================================================================

def flatten_to_csv(df, output_path="data/sp500_prices_raw.csv"):
    """
    Transform the MultiIndex DataFrame produced by yfinance into a
    single flat table with columns:

    Date, Open, High, Low, Close, Adj Close, Volume, Ticker

    Flattening is necessary because most downstream econometric routines
    expect a tabular dataset rather than a hierarchical structure.

    Returns
    -------
    DataFrame
        The flattened dataset.
    """
    print("Flattening data...")

    clean_rows = []
    for ticker in df.columns.levels[0]:
        sub = df[ticker].copy()
        sub["Ticker"] = ticker
        clean_rows.append(sub)

    flat = pd.concat(clean_rows)
    flat = flat.reset_index()  # Ensure Date is a column rather than an index

    flat.to_csv(output_path, index=False)
    print(f"Saved RAW flattened dataset to {output_path}")

    return flat



# =====================================================================
# 4. Cleaning routine: handle invalid data, duplicates, and types
# =====================================================================

def clean_prices(df):
    """
    Apply a set of cleaning steps to the flattened dataset.

    Cleaning operations include:
    - Parsing and validating dates.
    - Ensuring numeric price fields (non-numeric values become NaN and are removed).
    - Removing rows with missing critical variables.
    - Removing duplicated rows.
    - Excluding entries with invalid prices (e.g., zero close price).
    - Sorting the dataset for consistency and reproducibility.

    This step is essential for obtaining a well-behaved dataset suitable for:
    - Covariance matrix estimation
    - Factor model regression
    - Portfolio optimization
    """
    print("Cleaning dataset...")

    # Convert date column to datetime format
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Enforce numeric columns
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop observations missing essential information
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"])

    # Remove duplicated rows
    df = df.drop_duplicates()

    # Exclude logically invalid entries (e.g., zero closing price)
    df = df[df["Close"] > 0]

    # Sort for consistent ordering
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    print("Cleaning complete")
    return df



# =====================================================================
# 5. Data validation and quality summary
# =====================================================================

def validate_prices(df):
    """
    Produce a basic diagnostic summary of the cleaned dataset.

    This function does not modify the data; instead, it verifies key properties:
    - Total number of observations
    - Number of unique tickers
    - Remaining missing values
    - Number of duplicated rows

    Such diagnostics are standard in empirical finance to ensure
    the dataset is appropriate for econometric analysis.
    """
    summary = {
        "n_rows": len(df),
        "n_tickers": df["Ticker"].nunique(),
        "missing_values": df.isna().sum().sum(),
        "duplicates": df.duplicated().sum(),
    }

    print("Validation summary:")
    for k, v in summary.items():
        print(f" - {k}: {v}")

    return summary



# =====================================================================
# 6. Convenience wrapper: full pipeline execution
# =====================================================================

def full_download(output_path="data/sp500_prices_clean.csv", period="5y"):
    """
    Execute the full data ingestion pipeline:

    1. Retrieve S&P 500 constituent tickers.
    2. Download historical price data from Yahoo Finance.
    3. Flatten the MultiIndex structure.
    4. Clean the resulting dataset.
    5. Validate data integrity.
    6. Save cleaned data to CSV.

    Parameters
    ----------
    output_path : str
        Destination path for the cleaned dataset.
    period : str
        Historical horizon for price retrieval.
    """

    # Official valid periods as defined by yfinance API
    allowed_periods = {
        "1d", "5d", "1mo", "3mo", "6mo",
        "1y", "2y", "5y", "10y", "ytd", "max"
    }

    if period not in allowed_periods:
        print(f"Warning: period '{period}' is not officially in yfinance defaults.")
        print(f"Valid options include: {allowed_periods}")

    print("Starting FULL download pipeline...")

    tickers = get_sp500_tickers()
    raw_multi = download_prices(tickers, period=period)
    flat_raw = flatten_to_csv(raw_multi, output_path="sp500_prices_raw.csv")
    cleaned = clean_prices(flat_raw)
    validate_prices(cleaned)

    cleaned.to_csv(output_path, index=False)
    print(f"CLEAN dataset saved to: {output_path}")

    return cleaned

# =====================================================================
# 7. Script entry point
# =====================================================================

if __name__ == "__main__":
    # Default execution: download 5 years of data
    full_download(period="5y")
