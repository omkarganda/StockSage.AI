from typing import Optional

import pandas as pd

from .validation import DataValidator, validate_market_data
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Instantiate one shared validator (can be reused)
_validator: Optional[DataValidator] = None

def _get_validator() -> DataValidator:
    """Lazily create a DataValidator instance (singleton style)."""
    global _validator
    if _validator is None:
        _validator = DataValidator(strict_mode=False)
    return _validator


def clean_market_data(df: pd.DataFrame, symbol: str = "dataset", *, validate: bool = True) -> pd.DataFrame:
    """Perform standard cleaning steps for raw market data.

    Steps applied (in order):
    1. Ensure DatetimeIndex, sorted ascending.
    2. Drop duplicate timestamps (keep first occurrence).
    3. Drop rows containing only NaNs.
    4. Forward-fill then backward-fill remaining missing values.
    5. OPTIONAL – run StockSage's DataValidator and log a summary.

    Parameters
    ----------
    df : pd.DataFrame
        Raw market OHLCV dataframe.
    symbol : str, default "dataset"
        Symbol name (for logging / reporting only).
    validate : bool, default True
        Whether to run the full validation suite after cleaning.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for downstream processing.
    """
    if df.empty:
        logger.warning("clean_market_data received an empty DataFrame – nothing to clean.")
        return df

    # 1) Index handling — coerce to datetime & sort
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()

    # 2) Remove duplicate datetime rows (keeps first to preserve earliest quote)
    dup_count = df.index.duplicated().sum()
    if dup_count > 0:
        logger.info(f"Removed {dup_count} duplicate timestamp rows from {symbol} data")
        df = df[~df.index.duplicated(keep="first")]

    # 3) Drop fully-empty rows
    empty_rows = df.isnull().all(axis=1).sum()
    if empty_rows:
        logger.info(f"Dropped {empty_rows} completely empty rows from {symbol} data")
        df = df.dropna(how="all")

    # 4) Basic missing-value imputation – ffill then bfill
    df = df.ffill().bfill()

    # 5) Optional validation & logging
    if validate:
        validator = _get_validator()
        report = validator.validate_dataset(df, dataset_type="market", dataset_name=f"{symbol}_cleaned_market_data")
        report.print_summary()
        if report.has_critical_issues():
            logger.warning("clean_market_data: critical issues remain after cleaning – downstream steps should handle accordingly.")

    return df