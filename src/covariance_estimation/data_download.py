"""
Function to download financial time series data from Yahoo Finance and save it into a CSV file:

- Based on the S&P 500 Index composition, tickers are automatically retrieved from Wikipedia
- Tickers are used to download daily price data using the yfinance library
- The downloaded data is cleaned and saved into a CSV format
"""

import pandas as pd
import yfinance as yf
import requests


def get_sp500_tickers():
    """
    Retrieves the list of S&P 500 company tickers from Wikipedia.

    Returns:
        A list of ticker symbols as strings
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    return df["Symbol"].tolist()


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
    print(f"Downloading data for {len(tickers)} tickers...")
    df = yf.download(tickers, period=period, group_by="ticker", auto_adjust=False)
    return df


def flatten_to_csv(df, output_path="sp500_prices.csv"):
    """
    Converts the MultiIndex DataFrame obtained into a flat DataFrame
    with columns: Date, Open, High, Low, Close, Volume, Ticker

    Arguments:
    df : MultiIndex DataFrame returned by yfinance
    output_path : str
        File name to save the CSV

    Returns:
        flat DataFrame
    """
    print("Flattening data...")

    clean_rows = []
    for ticker in df.columns.levels[0]:
        sub = df[ticker].copy()
        sub["Ticker"] = ticker
        clean_rows.append(sub)

    flat = pd.concat(clean_rows)
    # Move Date from index to column
    flat = flat.reset_index()  
    flat.to_csv(output_path, index=False)

    print(f"Saved cleaned dataset to {output_path}")
    return flat


def validate_prices(df):
    """
    Identifies possible issues on the downloaded dataset:
    
    Arguments:
    df : DataFrame
        Flattened DataFrame from `flatten_to_csv()`.

    Returns:
    dict: summary of detected issues.
    """
    summary = {
        "n_rows": len(df),
        "n_tickers": df["Ticker"].nunique(),
        "missing_values": df.isna().sum().sum(),
        "duplicates": df.duplicated().sum(),
    }

    print("Validation summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


    def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean price dataframe to be ready to use:

    - Convert price columns to numeric
    - Ensure Date is a datetime and sorted
    - Remove duplicate rows, rows with NA values and non-positive prices
    """

    df = df.copy()

    #Date column is datetime
    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    #Sort by date
    df = df.sort_values("Date")

    #Convert price columns to numeric
    price_cols = ["Open", "High", "Low", "Close"]

    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    #Remove duplicated rows
    df = df.drop_duplicates()

    #Remove NA rows
    df = df.dropna(subset=["Date"] + price_cols)

    #Remove non-positive prices
    for col in price_cols:
        df = df[df[col] > 0]

    #Reset index after cleaning
    df = df.reset_index(drop=True)

    return df


def full_download(output_path="sp500_prices.csv"):
    """
    Convenience wrapper:
    1. Get S&P500 tickers
    2. Download 5 years of daily price data
    3. Flatten and save to CSV
    4. Validate the result

    Parameters
    ----------
    output_path : str
        Destination CSV file.

    Returns
    -------
    DataFrame
        Final cleaned DataFrame ready for analysis.
    """
    tickers = get_sp500_tickers()
    raw = download_prices(tickers, period="5y")
    flat = flatten_to_csv(raw, output_path=output_path)
    validate_prices(flat)
    return flat
