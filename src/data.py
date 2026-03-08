"""Data ingestion module — yfinance historical OHLCV fetcher."""
import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker: str, period: str = "2y") -> pd.Series:
    """Fetch daily closing prices for a given ticker.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        period: Lookback period string (e.g., "2y", "1y", "6mo")

    Returns:
        pd.Series of closing prices indexed by DatetimeIndex.

    Raises:
        ValueError: If ticker is invalid or returns no data.
    """
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    except Exception as e:
        raise ValueError(f"Failed to download data for ticker={ticker!r}: {e}") from e

    if df.empty:
        raise ValueError(f"No data returned for ticker={ticker!r}. Check the symbol.")

    close = df["Close"].squeeze()
    close.name = ticker
    return close
