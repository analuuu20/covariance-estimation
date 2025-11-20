"""
data_split.py

This module performs a chronological time-series split for a panel dataset
containing multiple equities stacked vertically. Each ticker's price history is
split individually into a training and validation set to ensure methodological
correctness and avoid cross-sectional leakage.

The script loads the cleaned dataset from:
    data/sp500_prices_clean.csv

It outputs:
    data/train_prices.csv
    data/validation_prices.csv
"""

import pandas as pd
import os


def chronological_panel_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Perform a chronological split for each ticker in the panel dataset.

    Academic justification:
    -----------------------
    Financial time-series must be split chronologically because future data
    cannot be used to inform past model estimation (Hyndman & Athanasopoulos, 2018).
    For panel equity data, treating the dataset as a simple row-wise sequence
    is incorrect because tickers differ in history length and ordering. Therefore,
    each ticker must be split independently, maintaining temporal integrity.

    Parameters
    ----------
    df : pd.DataFrame
        Panel dataset with at least columns ["Date", "Ticker"].
    train_ratio : float
        Proportion of each ticker’s time span assigned to training.

    Returns
    -------
    train_df : pd.DataFrame
        Training dataset with chronological splits per ticker.
    val_df : pd.DataFrame
        Validation dataset with chronological splits per ticker.
    """

    # Ensure proper ordering
    df = df.sort_values(["Ticker", "Date"])

    train_list = []
    val_list = []

    # Split per ticker
    for ticker, group in df.groupby("Ticker"):
        n = len(group)
        train_cutoff = int(n * train_ratio)

        train_part = group.iloc[:train_cutoff]
        val_part = group.iloc[train_cutoff:]

        train_list.append(train_part)
        val_list.append(val_part)

    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)

    return train_df, val_df


def validate_split(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Validate that the split is methodologically correct.

    This function performs:
    1. Chronology check (max(train_date) < min(val_date)).
    2. No-leakage check (no overlapping dates).
    3. Consistency check (every ticker appears in both sets unless too short).

    Prints validation results.
    """

    print("\n===== VALIDATION REPORT =====\n")

    tickers = sorted(set(train_df["Ticker"]).union(val_df["Ticker"]))

    for ticker in tickers:
        train_part = train_df[train_df["Ticker"] == ticker]
        val_part = val_df[val_df["Ticker"] == ticker]

        if len(train_part) == 0 or len(val_part) == 0:
            print(f"[WARNING] Ticker {ticker} does not appear in both splits.")
            continue

        last_train = train_part["Date"].max()
        first_val = val_part["Date"].min()

        if last_train < first_val:
            print(f"[OK] {ticker}: chronological integrity preserved.")
        else:
            print(f"[ERROR] {ticker}: chronology violated!")

        overlap = set(train_part["Date"]).intersection(val_part["Date"])
        if overlap:
            print(f"[ERROR] {ticker}: date overlap detected!")
        else:
            print(f"[OK] {ticker}: no leakage.")


if __name__ == "__main__":
    # Load dataset
    input_path = "data/sp500_prices_clean.csv"
    df = pd.read_csv(input_path, parse_dates=["Date"])

    # Perform split
    train_df, val_df = chronological_panel_split(df, train_ratio=0.8)

    # Save outputs
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train_prices.csv", index=False)
    val_df.to_csv("data/validation_prices.csv", index=False)

    print("Training and validation sets saved successfully.")

    # Run validation checks
    validate_split(train_df, val_df)
