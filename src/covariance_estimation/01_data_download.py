"""
DATA DOWNLOAD AND CLEANING MODULE:

This module develops the necessary utility functions to download, transform, clean, and validate 
financial time series data that is used throughout the covariance estimation project.

The pipeline follows these steps:

1. Retrieve the list of S&P 500 constituents from Wikipedia.
2. Download historical OHLCV (Open, High, Low, Close, Adjusted Close, Volume)
   price data for each ticker using the yfinance API.
3. Convert the MultiIndex structure produced by yfinance into a flat, tabular dataset.
4. Clean the dataset by removing invalid values, duplicates, and non-numeric entries.
5. Validate the resulting dataset using basic quality checks.
6. Save the cleaned dataset into CSV format for further analysis in the project.

"""

import pandas as pd
import yfinance as yf
import requests


# =====================================================================
# 1. Retrieve S&P 500 tickers from Wikipedia
# =====================================================================

def get_sp500_tickers():
    """
    Extract the S&P 500 constituent list directly from Wikipedia.

    This function:
    - Uses an explicit User-Agent to avoid server-side blocking.
    - Parses all HTML tables on the page.
    - Selects the table that contains the 'Symbol' column.
    - Converts tickers to Yahoo Finance format (replace '.' with '-').

    Returns
    -------
    list[str]
        Clean list of S&P 500 ticker symbols.
    """

    from io import StringIO
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch S&P 500 list. Status code: {response.status_code}")

    dfs = pd.read_html(StringIO(response.text))

    table = None
    for df in dfs:
        if "Symbol" in df.columns:
            table = df
            break

    if table is None:
        raise ValueError("Could not find S&P 500 table containing the 'Symbol' column.")

    tickers = [t.replace(".", "-") for t in table["Symbol"].tolist()]

    print("\n Total S&P 500 tickers retrieved:", len(tickers))
    print("Tickers used:\n", tickers)

    return tickers



# =====================================================================
# 2. Download price data using yfinance 
# =====================================================================

def download_prices(tickers, start_date, end_date):
    """
    Download Open, High, Low, Close, Adjusted Close, Volume price data for all tickers within the specified date range.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    DataFrame
        MultiIndex DataFrame with (Variable, Ticker) columns.
    """

    print(f"\n Downloading data for {len(tickers)} tickers")
    print(f"           From: {start_date}  To: {end_date}")

    df = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        group_by="ticker",
        auto_adjust=False
    )

    print("[DOWNLOAD] Completed download.\n")
    return df



# =====================================================================
# 3. Flatten MultiIndex DataFrame to a simple table
# =====================================================================

def flatten_to_csv(df, output_path="data/sp500_prices_raw.csv"):
    """
    Convert the MultiIndex DataFrame from yfinance into a flat structure,
    with columns:

        Date, Open, High, Low, Close, Adj Close, Volume, Ticker

    Returns
    -------
    DataFrame
        Flattened price dataset.
    """

    print("[FLATTEN] Transforming MultiIndex dataset into tabular format...")

    clean_rows = []
    for ticker in df.columns.levels[0]:
        sub = df[ticker].copy()
        sub["Ticker"] = ticker
        clean_rows.append(sub)

    flat = pd.concat(clean_rows)
    flat = flat.reset_index()

    flat.to_csv(output_path, index=False)
    print(f"[FLATTEN] Saved RAW flattened dataset -> {output_path}")

    return flat



# =====================================================================
# 4. Cleaning routine
# =====================================================================

def clean_prices(df):
    """
    Applies standard data cleaning procedures commonly used in empirical finance.

    These include:
    - Date parsing
    - Enforcing numeric types
    - Removing missing values and duplicates
    - Filtering out invalid price entries
    - Sorting the dataset
    """

    print("[CLEAN] Cleaning dataset...")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"])
    df = df.drop_duplicates()
    df = df[df["Close"] > 0]
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    print("[CLEAN] Cleaning complete.\n")
    return df



# =====================================================================
# 5. Validation summary
# =====================================================================

def validate_prices(df):
    """
    Generates a diagnostic summary for data validation after cleaning.
    """

    summary = {
        "n_rows": len(df),
        "n_tickers": df["Ticker"].nunique(),
        "missing_values": df.isna().sum().sum(),
        "duplicates": df.duplicated().sum(),
    }

    print("[VALIDATION] Dataset statistics:")
    for k, v in summary.items():
        print(f" - {k}: {v}")

    return summary



# =====================================================================
# 6. Full pipeline for data download and processing
# =====================================================================

def full_download(output_path="data/sp500_prices_clean.csv"):
    """
    Full data ingestion workflow with fixed date range:
    15 Nov 2020 -> 14 Nov 2025
    """

    start = "2020-11-15"
    end   = "2025-11-14"

    print("\n========== FULL DATA DOWNLOAD PIPELINE ==========\n")
    print(f"[INFO] Using fixed date range: {start} → {end}")

    tickers = get_sp500_tickers()
    raw_multi = download_prices(tickers, start, end)
    flat_raw = flatten_to_csv(raw_multi, output_path="data/sp500_prices_raw.csv")
    cleaned = clean_prices(flat_raw)
    validate_prices(cleaned)

    cleaned.to_csv(output_path, index=False)
    print(f"\n[SAVE] CLEAN dataset saved to: {output_path}")

    print("\n========== DATA DOWNLOAD AND CLEANING COMPLETED SUCCESSFULLY ==========\n")
    return cleaned



# ---------------------------------------------------------------------
# 7. MAIN EXECUTION
# ---------------------------------------------------------------------

if __name__ == "__main__":
    full_download()
