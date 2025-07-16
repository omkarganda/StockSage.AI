from __future__ import annotations

"""Simple vectorised backtesting engine

This lightweight engine provides the minimum functionality required by the
StockSage.AI roadmap:

• Walk-forward simulation of position signals (long/flat/short).
• Portfolio-level equity curve starting from an initial capital.
• Calculation of common performance statistics (cumulative return, annualised
  Sharpe ratio, maximum drawdown).

It purposefully avoids heavyweight libraries such as *backtrader* or
*vectorbt* so that users can run back-tests without additional dependencies.
"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd

from ..config import ModelConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)

Signal = Literal["buy", "sell", "hold", "long", "short", "flat"]


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]


class Backtester:
    """A vectorised long-only / long-short back-tester for daily data."""

    def __init__(
        self,
        initial_capital: float | None = None,
        commission_rate: float | None = None,
        slippage_rate: float | None = None,
    ) -> None:
        cfg = ModelConfig
        self.initial_capital = initial_capital or cfg.INITIAL_CAPITAL
        self.commission_rate = commission_rate if commission_rate is not None else cfg.COMMISSION_RATE
        self.slippage_rate = slippage_rate if slippage_rate is not None else cfg.SLIPPAGE_RATE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        price_series: pd.Series,
        signal_series: pd.Series,
    ) -> BacktestResult:
        """Run a back-test and return the equity curve & trade log.

        Parameters
        ----------
        price_series : pd.Series
            Close prices indexed by date.
        signal_series : pd.Series
            Signal per day. Supported values: "buy"/"sell"/"hold" or
            numerical weights in the range [-1, 1]. The index **must** align
            with `price_series`.
        """
        price_series = price_series.astype(float)
        signal_series = _standardise_signals(signal_series)

        if not price_series.index.equals(signal_series.index):
            raise ValueError("Price and signal indices must be identical and aligned.")

        # ------------------------------------------------------------------
        # Generate positions (next-day open execution assumption)
        # ------------------------------------------------------------------
        positions = signal_series.shift(1).fillna(0)  # enter trade on next day

        # Portfolio daily returns with commissions & slippage
        pct_change = price_series.pct_change().fillna(0.0)
        gross_returns = positions * pct_change

        # Transaction costs on |Δposition|
        position_change = positions.diff().abs().fillna(0.0)
        costs = position_change * self.commission_rate + position_change * self.slippage_rate

        net_returns = gross_returns - costs

        equity_curve = (1 + net_returns).cumprod() * self.initial_capital

        # ------------------------------------------------------------------
        # Trades log (entry/exit times, pnl)
        # ------------------------------------------------------------------
        trades = _extract_trades(positions, price_series)

        # Metrics
        metrics = _compute_metrics(equity_curve, net_returns)

        return BacktestResult(equity_curve=equity_curve, trades=trades, metrics=metrics)

    # ------------------------------------------------------------------
    # Parameter sweep helper
    # ------------------------------------------------------------------
    def run_parameter_sweep(
        self,
        price_series: pd.Series,
        strategy_func,
        param_grid: Dict[str, list],
    ) -> pd.DataFrame:
        """Run back-tests for each combination in the param grid.

        Parameters
        ----------
        strategy_func  : callable(prices: pd.Series, **params) -> pd.Series
            Function that generates signal series given prices and params.
        param_grid     : Dict[str, list]
            Parameter grid like {"short_window": [5, 10], "long_window": [20, 50]}.
        Returns
        -------
        DataFrame with metrics per parameter combination.
        """
        import itertools

        keys = list(param_grid.keys())
        rows = []
        for values in itertools.product(*[param_grid[k] for k in keys]):
            params = dict(zip(keys, values))
            try:
                signals = strategy_func(price_series, **params)
                res = self.run(price_series, signals)
                metric_row = {**params, **res.metrics}
                rows.append(metric_row)
            except Exception as exc:
                logger.warning("Sweep failed for params %s – %s", params, exc)
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def _standardise_signals(sig: pd.Series) -> pd.Series:
    mapping: Dict[str, float] = {
        "buy": 1.0,
        "long": 1.0,
        "sell": -1.0,
        "short": -1.0,
        "hold": 0.0,
        "flat": 0.0,
    }
    if sig.dtype == object:
        return sig.str.lower().map(mapping).fillna(0.0)
    return sig.clip(-1.0, 1.0)


def _extract_trades(positions: pd.Series, prices: pd.Series) -> pd.DataFrame:
    """Return a dataframe with trade entry/exit times and PnL."""
    pos_change = positions.diff().fillna(positions.iloc[0])

    entries = pos_change[pos_change != 0]
    trade_records = []
    current_entry_price: Optional[float] = None
    current_direction: float = 0.0
    entry_date: Optional[pd.Timestamp] = None

    for date, change in entries.items():
        direction = positions.loc[date]
        if direction != 0 and current_direction == 0:  # Open trade
            current_entry_price = prices.loc[date]
            current_direction = direction
            entry_date = date
        elif direction == 0 and current_direction != 0:  # Close trade
            exit_price = prices.loc[date]
            pnl = (exit_price - current_entry_price) * current_direction
            trade_records.append(
                {
                    "entry_date": entry_date,
                    "exit_date": date,
                    "direction": "long" if current_direction > 0 else "short",
                    "entry_price": current_entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                }
            )
            current_entry_price = None
            current_direction = 0.0
            entry_date = None

    return pd.DataFrame(trade_records)


def _compute_metrics(equity_curve: pd.Series, daily_returns: pd.Series) -> Dict[str, float]:
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    ann_return = (1 + total_return) ** (252 / len(equity_curve)) - 1 if len(equity_curve) > 1 else 0.0

    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0.0

    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()

    # Sortino ratio (downside deviation)
    downside = daily_returns.clip(upper=0)
    downside_std = downside.std() * np.sqrt(252)
    sortino = ann_return / abs(downside_std) if downside_std != 0 else 0.0

    # Calmar ratio
    calmar = (-ann_return / max_dd) if max_dd < 0 else 0.0

    # Win rate
    win_rate = (daily_returns > 0).mean()

    # Trade statistics (assuming positions changes define trades)
    trade_count = (daily_returns != 0).sum()

    return {
        "cumulative_return": float(total_return),
        "annualised_return": float(ann_return),
        "annualised_volatility": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "sortino_ratio": float(sortino),
        "calmar_ratio": float(calmar),
        "win_rate": float(win_rate),
        "trading_days": int(len(equity_curve)),
        "trade_count": int(trade_count),
    }