#!/usr/bin/env python
"""Run quick back-test for a single symbol using StockSage predictions.

Example
-------
$ python scripts/backtest.py --symbol AAPL --start 2023-01-01 --end 2023-12-31

The script fetches historical prices via the existing MarketDataDownloader,
generates simple momentum-based signals *or* uses predictions from the API if
available, and then runs the vectorised Backtester.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.download_market import MarketDataDownloader
from src.backtesting import Backtester
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _generate_signals(prices: pd.Series) -> pd.Series:
    """Very simple moving-average crossover signal used as default placeholder."""
    short_ma = prices.rolling(window=5).mean()
    long_ma = prices.rolling(window=20).mean()

    signal = pd.Series(index=prices.index, data="hold")
    signal = signal.where(short_ma <= long_ma, "buy")
    signal = signal.where(short_ma >= long_ma, signal)  # keep previous for equality
    return signal


def main():
    parser = argparse.ArgumentParser(description="Run a quick back-test for a given symbol.")
    parser.add_argument("--symbol", required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--initial_cap", type=float, default=None, help="Initial capital")
    args = parser.parse_args()

    downloader = MarketDataDownloader(use_cache=True)
    df = downloader.download_stock_data(args.symbol, args.start, args.end)
    if df is None or df.empty:
        logger.error("No price data found for %s", args.symbol)
        return

    prices = df["Close"]
    signals = _generate_signals(prices)

    backtester = Backtester(initial_capital=args.initial_cap)
    result = backtester.run(prices, signals)

    logger.info("Back-test completed – metrics: %s", result.metrics)

    # Save equity curve for inspection
    out_dir = Path("results/backtests")
    out_dir.mkdir(parents=True, exist_ok=True)
    equity_path = out_dir / f"equity_{args.symbol}_{args.start}_{args.end}.csv"
    result.equity_curve.to_csv(equity_path, header=["equity"])
    logger.info("Equity curve saved to %s", equity_path)


if __name__ == "__main__":
    main()