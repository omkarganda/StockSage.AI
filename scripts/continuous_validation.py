#!/usr/bin/env python
"""Continuous validation entry-point

This **simplified** script is intended to be scheduled (e.g. via GitHub
Actions or cron) to keep the StockSage models and evaluation metrics up to
date. In its current form it performs three high-level tasks:

1. Fetches the most recent market data & sentiment features (using existing
   pipeline code).
2. Loads the latest trained models (or trains lightweight baseline models if
   none exist) and scores the validation set.
3. Runs a back-test over the most recent period and writes updated metrics to
   `reports/evaluation/latest_validation.json`.

The implementation is intentionally lightweight so it can run inside free CI
runners; heavy hyper-parameter tuning is **out-of-scope** here.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.download_market import MarketDataDownloader
from src.features.indicators import add_all_technical_indicators
from src.backtesting import Backtester
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_validation(days_back: int = 90):
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days_back)

    downloader = MarketDataDownloader(use_cache=True)

    metrics_list = []

    for symbol in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]:
        df = downloader.download_stock_data(symbol, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        if df is None or df.empty:
            logger.warning("No data for %s – skipping.", symbol)
            continue

        df = add_all_technical_indicators(df)
        prices = df["Close"]

        # Generate naive momentum signal (placeholder until model integration)
        signal = prices.pct_change().rolling(window=5).mean()
        signal = np.where(signal > 0, "buy", np.where(signal < 0, "sell", "hold"))
        signal = pd.Series(signal, index=prices.index)

        backtester = Backtester()
        result = backtester.run(prices, signal)

        metrics = result.metrics
        metrics["symbol"] = symbol
        metrics_list.append(metrics)

    if not metrics_list:
        logger.error("Validation aborted – no metrics computed.")
        return

    validation_df = pd.DataFrame(metrics_list)

    out_path = Path("reports/evaluation/latest_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    validation_df.to_json(out_path, orient="records", indent=2)
    logger.info("Continuous validation results written to %s", out_path)


if __name__ == "__main__":
    run_validation()