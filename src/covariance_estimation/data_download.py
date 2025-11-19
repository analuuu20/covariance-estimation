"""
Function to download financial time series data from Yahoo Finance and save it into a CSV file:

- Based on the S&P 500 Index composition, tickers are automatically retrieved from Wikipedia
- Tickers are used to download daily price data using the yfinance library
- The downloaded data is cleaned and saved into a CSV format
"""

import pandas as pd
import yfinance as yf
import requests

# tickers downloaded: S&P 500 from Wikipedia

def get_sp500_tickers():
    """
    Extracts S&P 500 tickers from Wikipedia with robust request    
    and robust table selection (to avoid errors due diferent notation in Wikipedia).
    """
    
    from io import StringIO

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/123.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch S&P 500 list. Status code: {response.status_code}")

    # Avoid FutureWarning by wrapping response.text in StringIO
    dfs = pd.read_html(StringIO(response.text))

    # Find the table that contains the 'Symbol' column
    table = None
    for df in dfs:
        if "Symbol" in df.columns:
            table = df
            break

    if table is None:
        raise ValueError("Could not find S&P 500 table with 'Symbol' column.")

    tickers = table["Symbol"].tolist()
    tickers = [t.replace(".", "-") for t in tickers]  # Yahoo compatibility

    return tickers



# price data downloaded from yfinance

def download_prices(tickers, period="5y"):
    """
    Downloads open, high, low, close daily price and volume of trading for the list of tickers retrieved previously using yfinance.

    Arguments:
    tickers : list[str]
        List of ticker symbols
    period : str refering the desired time period ('1y', '5y', 'max')

    Returns:
    DataFrame
        A MultiIndex DataFrame with columns:
        ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        and tickers as the second level of the column index.
    """
    print(f"Downloading data for {len(tickers)} tickers over period = '{period}'...")
    df = yf.download(tickers, period=period, group_by="ticker", auto_adjust=False)
    return df

# transform multiindex dataframe to flat table

def flatten_to_csv(df, output_path="data/sp500_prices_raw.csv"):
    """
    Converts yfinance's MultiIndex DataFrame into a clean table with this columns:
    Date, Open, High, Low, Close, Adj Close, Volume, Ticker
    """
    print("Flattening data...")

    clean_rows = []
    for ticker in df.columns.levels[0]:
        sub = df[ticker].copy()
        sub["Ticker"] = ticker
        clean_rows.append(sub)

    flat = pd.concat(clean_rows)
    flat = flat.reset_index()  # Move Date out of index

    flat.to_csv(output_path, index=False)
    print(f"Saved RAW flattened dataset to {output_path}")

    return flat

# data cleaning: remove NaNs, duplicates, invalid prices and ensure correct types of data

def clean_prices(df):
    """
    Cleans the flattened dataset:

    - Remove duplicates
    - Drop rows with NaNs
    - Ensure numeric prices
    - Ensure valid dates
    - Remove tickers with no data
    - Sort data by Ticker and Date
    """

    print("Cleaning dataset...")

    # Convert date column
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Convert numeric columns
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"])

    # Drop duplicated rows
    df = df.drop_duplicates()

    # Drop tickers with zero prices
    df = df[df["Close"] > 0]

    # Sort the final cleaned dataset
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    print("Cleaning complete")
    return df

# data validation: check data quality on the new cleaned dataset
def validate_prices(df):
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


# Convenience wrapper to run the full pipeline

def full_download(output_path="data/sp500_prices_clean.csv", period="5y"):
    """
    Full pipeline:
    1. Get S&P500 tickers
    2. Download open, high, low, close daily price and volume of trading for a chosen period
    3. Flatten MultiIndex data to a flat table
    4. CLEAN the flat raw dataset
    5. Validate the cleaned dataset
    6. Save clean dataset into a CSV file

    Arguments:
    output_path : str
        Name to save the cleaned CSV.
    period : str
        Time span of historical data (defined allowed periods supported by yfinance).
    
    """

    allowed_periods = {"1d", "5d", "1mo", "3mo", "6mo",
                       "1y", "2y", "5y", "10y", "ytd", "max"}

    if period not in allowed_periods:
        print(f"Warning: period '{period}' is not officially in yfinance defaults.")
        print(f"Using it anyway. Valid options include: {allowed_periods}")

    print("Starting FULL download pipeline...")

    tickers = get_sp500_tickers()
    raw_multi = download_prices(tickers, period=period)
    flat_raw = flatten_to_csv(raw_multi, output_path="sp500_prices_raw.csv")
    cleaned = clean_prices(flat_raw)
    validate_prices(cleaned)

    cleaned.to_csv(output_path, index=False)
    print(f"CLEAN dataset saved to: {output_path}")

    return cleaned

# to run the full pipeline when executing this script

if __name__ == "__main__":
    #run by default: 5 years period
    full_download(period="5y")
