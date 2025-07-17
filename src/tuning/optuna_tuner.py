import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import optuna
import pandas as pd

from ..models.lstm_attention_model import LSTMAttentionModel
from ..models.transformer_model import TransformerModel
from ..utils.logging import get_logger

logger = get_logger(__name__)


# ------------------------- generic utils -----------------------------


def _train_val_split(df: pd.DataFrame, val_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - val_frac))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


# ------------------------- LSTM-Attention tuning ---------------------

def tune_lstm_attention(
    df: pd.DataFrame,
    context_length: int,
    horizon: int,
    n_trials: int = 25,
    study_name: str | None = None,
    output_dir: str | Path = "results/hparam_tuning",
    device: str = "auto",
) -> Dict[str, float]:
    """Run Optuna study to tune the LSTM-Attention model.

    Returns the best hyperparameter dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    study_name = study_name or "lstm_attention_tuning"
    study_path = output_dir / f"{study_name}.db"

    def objective(trial: optuna.Trial):
        hidden_size = trial.suggest_int("hidden_size", 32, 256, log=True)
        num_layers = trial.suggest_int("num_layers", 1, 4)
        attention_heads = trial.suggest_int("attention_heads", 1, 8)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        epochs = trial.suggest_int("epochs", 3, 8)

        model = LSTMAttentionModel(
            context_length=context_length,
            horizon=horizon,
            hidden_size=hidden_size,
            num_layers=num_layers,
            attention_heads=attention_heads,
            lr=lr,
            epochs=epochs,
            batch_size=64,
            device=device,
        )

        train_df, val_df = _train_val_split(df, val_frac=0.2)
        try:
            model.fit(train_df)
            metrics = model.evaluate(val_df)
            return metrics.get("mse", np.inf)
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return np.inf

    study = optuna.create_study(direction="minimize", study_name=study_name,
                                storage=f"sqlite:///{study_path}", load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"LSTM-Attention tuning completed. Best MSE: {study.best_value:.6f}")
    logger.info(f"Best params: {study.best_params}")

    # Save best params to JSON
    with open(output_dir / f"{study_name}_best.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

    return study.best_params


# ------------------------- Transformer tuning ------------------------

def tune_transformer(
    df: pd.DataFrame,
    context_length: int,
    horizon: int,
    n_trials: int = 25,
    study_name: str | None = None,
    output_dir: str | Path = "results/hparam_tuning",
    device: str = "auto",
) -> Dict[str, float]:
    study_name = study_name or "transformer_tuning"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    study_path = output_dir / f"{study_name}.db"

    def objective(trial: optuna.Trial):
        d_model = trial.suggest_int("d_model", 32, 128, step=32)
        num_layers = trial.suggest_int("num_layers", 1, 4)
        nhead = trial.suggest_int("nhead", 2, 8)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.05, 0.3)
        epochs = trial.suggest_int("epochs", 3, 8)

        model = TransformerModel(
            context_length=context_length,
            horizon=horizon,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            epochs=epochs,
            batch_size=64,
            device=device,
        )

        train_df, val_df = _train_val_split(df, 0.2)
        try:
            model.fit(train_df)
            metrics = model.evaluate(val_df)
            return metrics.get("mse", np.inf)
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return np.inf

    study = optuna.create_study(direction="minimize", study_name=study_name,
                                storage=f"sqlite:///{study_path}", load_if_exists=True)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"Transformer tuning completed. Best MSE: {study.best_value:.6f}")
    logger.info(f"Best params: {study.best_params}")
    with open(output_dir / f"{study_name}_best.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

    return study.best_params