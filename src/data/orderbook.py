import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "load_orderbook_csv",
    "aggregate_orderbook_levels",
]


def load_orderbook_csv(file_path: str | Path, *, depth: int = 1, tz: Optional[str] = "UTC") -> pd.DataFrame:
    """Load order-book snapshot CSV and return a cleaned DataFrame.

    Expected CSV schema (typical for Level-2 snapshots):
    timestamp,bid_price_1,bid_size_1,ask_price_1,ask_size_1,bid_price_2,...

    The function:
    • parses `timestamp` to datetime index (timezone-aware)
    • keeps only the top `depth` levels (default 1)
    • renames the best-bid/ask columns to standardised names used across project:
        bid_price, bid_size, ask_price, ask_size

    Parameters
    ----------
    file_path : str | pathlib.Path
        Path to CSV file.
    depth : int, default 1
        Number of book levels to keep (starting from best bid/ask).
    tz : str or None
        Target timezone for timestamp column (None = naive).
    """
    file_path = Path(file_path)
    logger.info(f"Loading order-book CSV: {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    df = pd.read_csv(file_path)
    if "timestamp" not in df.columns:
        raise ValueError("CSV must contain a 'timestamp' column")

    # Parse timestamp & set index
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if tz:
        df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
    df = df.set_index("timestamp").sort_index()

    # Determine columns for requested depth
    cols_to_keep: List[str] = []
    for level in range(1, depth + 1):
        for side in ("bid", "ask"):
            for field in ("price", "size"):
                pattern = f"{side}_{field}_{level}"
                matches = [c for c in df.columns if pattern.lower() == c.lower()]
                if matches:
                    cols_to_keep.append(matches[0])

    if not cols_to_keep:
        raise ValueError("Could not find matching bid/ask columns – check CSV schema")

    df = df[cols_to_keep]

    # Rename level-1 columns to canonical names
    rename_map = {}
    for side in ("bid", "ask"):
        for field in ("price", "size"):
            lvl1 = f"{side}_{field}_1"
            for col in df.columns:
                if col.lower() == lvl1:
                    rename_map[col] = f"{side}_{field}"
    df = df.rename(columns=rename_map)

    logger.info(f"Loaded order-book snapshots: {df.shape[0]:,} rows, depth={depth}")
    return df


def aggregate_orderbook_levels(df: pd.DataFrame, depth: int = 5) -> pd.DataFrame:
    """Aggregate multiple book levels into summary statistics.

    Produces level-aware features such as volume-weighted mid-price and total depth.
    Adds columns:
      • `ob_mid_price_wap` – WAP across *depth* levels
      • `ob_total_volume`  – sum of bid & ask size levels
      • `ob_spread_wap`    – (ask_wap – bid_wap) / mid
    """
    df = df.copy()
    bid_prices, bid_sizes, ask_prices, ask_sizes = [], [], [], []
    for lvl in range(1, depth + 1):
        for side, price_list, size_list in (
            ("bid", bid_prices, bid_sizes),
            ("ask", ask_prices, ask_sizes),
        ):
            p_col = f"{side}_price_{lvl}"
            s_col = f"{side}_size_{lvl}"
            if p_col in df.columns and s_col in df.columns:
                price_list.append(df[p_col])
                size_list.append(df[s_col])

    if bid_prices and ask_prices:
        bid_wap = sum(p * s for p, s in zip(bid_prices, bid_sizes)) / (sum(bid_sizes) + 1e-8)
        ask_wap = sum(p * s for p, s in zip(ask_prices, ask_sizes)) / (sum(ask_sizes) + 1e-8)
        mid = (bid_wap + ask_wap) / 2.0
        df["ob_mid_price_wap"] = mid
        df["ob_spread_wap"] = (ask_wap - bid_wap) / mid.replace(0, pd.NA)
        df["ob_total_volume"] = sum(bid_sizes) + sum(ask_sizes)
    else:
        logger.warning("aggregate_orderbook_levels: not all required columns present – skipping.")

    return df