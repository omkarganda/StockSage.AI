from typing import List

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


def _col_search(df: pd.DataFrame, keywords: List[str]) -> str:
    """Utility to find a column containing *all* keywords (case-insensitive)."""
    for col in df.columns:
        col_lower = col.lower()
        if all(k in col_lower for k in keywords):
            return col
    return ""


def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add common market microstructure & order-book derived features.

    The function is resilient to missing columns – it will add only the features
    that can be computed from the available data.

    Expected column patterns (case-insensitive):
    • bid_price / ask_price
    • bid_size / ask_size

    Added features (all prefixed `micro_`):
    1. `micro_mid_price` – (bid + ask)/2
    2. `micro_spread_pct` – (ask - bid) / mid
    3. `micro_quote_imbalance` – (bid_size - ask_size) / (bid_size + ask_size)
    4. `micro_mid_return` – log(mid_price).diff()

    Returns
    -------
    pd.DataFrame – original df with extra feature columns.
    """
    df = df.copy()

    bid_price_col = _col_search(df, ["bid", "price"])
    ask_price_col = _col_search(df, ["ask", "price"])

    if bid_price_col and ask_price_col:
        mid_price = (df[bid_price_col] + df[ask_price_col]) / 2.0
        df["micro_mid_price"] = mid_price
        df["micro_spread_pct"] = (df[ask_price_col] - df[bid_price_col]) / mid_price.replace(0, np.nan)
        # Return of mid price (log scale)
        df["micro_mid_return"] = np.log(mid_price).diff()
    else:
        logger.debug("Bid/Ask price columns not found – skipping mid_price & spread features")

    # Imbalance
    bid_size_col = _col_search(df, ["bid", "size"])
    ask_size_col = _col_search(df, ["ask", "size"])
    if bid_size_col and ask_size_col:
        imbalance = (df[bid_size_col] - df[ask_size_col]) / (
            df[bid_size_col] + df[ask_size_col] + 1e-8
        )
        df["micro_quote_imbalance"] = imbalance
    else:
        logger.debug("Bid/Ask size columns not found – skipping quote imbalance feature")

    return df