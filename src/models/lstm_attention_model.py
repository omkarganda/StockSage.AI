import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from ..features.indicators import add_all_technical_indicators
from ..features.microstructure import add_microstructure_features
from ..utils.logging import get_logger

logger = get_logger(__name__)

def _select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Return only numeric columns (no object/bool)."""
    return df.select_dtypes(include=[np.number])


class _SeqDataset(Dataset):
    """Simple torch dataset for sequence -> target mapping."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class _LSTMAttentionNet(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, attention_heads: int, horizon: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=attention_heads, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)  # Self-attention
        # Use last timestep representation
        last = attn_out[:, -1, :]
        out = self.fc(last)
        return out.squeeze(-1)


class LSTMAttentionModel:
    """LSTM + Self-Attention hybrid for financial forecasting (returns prediction)."""

    def __init__(
        self,
        context_length: int = 60,
        horizon: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        attention_heads: int = 4,
        epochs: int = 100,  # Increased from 50 to 100 for better training
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int = 15,  # Increased patience for early stopping
        min_delta: float = 1e-6,  # Minimum change to qualify as an improvement
        device: Optional[str] = None,
    ):
        # Ensure hidden_size is divisible by attention_heads
        if hidden_size % attention_heads != 0:
            # Adjust hidden_size to be divisible by attention_heads
            hidden_size = ((hidden_size // attention_heads) + 1) * attention_heads
            logger.info(f"Adjusted hidden_size to {hidden_size} to be divisible by {attention_heads} attention heads")
        
        self.context_length = context_length
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_heads = attention_heads
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.min_delta = min_delta
        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialized later
        self.model: Optional[nn.Module] = None
        self.scaler: Optional[StandardScaler] = None
        self.input_dim: Optional[int] = None
        self.is_fitted: bool = False
        self.training_metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Add technical + microstructure indicators (robust to missing columns)
        df = add_all_technical_indicators(df)
        df = add_microstructure_features(df)
        return _select_numeric(df).ffill().bfill()

    def _make_sequences(self, features: np.ndarray, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        seq_len = self.context_length
        horizon = self.horizon
        X_seqs, y_vals = [], []
        # Loop until there are `horizon` target values available
        for i in range(seq_len, len(features) - horizon + 1):
            X_seqs.append(features[i - seq_len : i])
            # Target = sequence of returns over the horizon, relative to the last price of the input sequence
            last_price = close[i - 1]
            future_prices = close[i : i + horizon]

            # Avoid division by zero
            if last_price > 1e-9:
                rets = (future_prices - last_price) / last_price
            else:
                rets = np.zeros_like(future_prices)

            y_vals.append(rets)
        return np.asarray(X_seqs, dtype=np.float32), np.asarray(y_vals, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "LSTMAttentionModel":
        logger.info(f"[LSTM-Attention] Training with {len(df)} rows")
        if "Close" not in df.columns:
            raise ValueError("Close column required for training")

        # Feature engineering
        feats_df = self._prepare_features(df)

        # Scale features
        self.scaler = StandardScaler()
        feats_scaled = self.scaler.fit_transform(feats_df.values)

        close_arr = df["Close"].values.astype(np.float32)
        X, y = self._make_sequences(feats_scaled, close_arr)
        if len(X) == 0:
            raise ValueError("Not enough data to build training sequences")

        dataset = _SeqDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.input_dim = X.shape[-1]
        assert self.input_dim is not None

        self.model = _LSTMAttentionNet(
            input_dim=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            attention_heads=self.attention_heads,
            horizon=self.horizon,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        
        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True
        )

        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        train_losses = []
        
        logger.info(f"Starting training for {self.epochs} epochs with patience {self.patience}")
        
        for epoch in range(1, self.epochs + 1):
            epoch_losses = []
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(xb)
                # Ensure pred and yb have compatible shapes for loss calculation
                if pred.dim() != yb.dim():
                    if pred.dim() == 1 and yb.dim() == 1:
                        pass  # Both are 1D, shapes should match
                    elif pred.dim() == 2 and pred.shape[1] == 1:
                        pred = pred.squeeze(-1)  # Remove last dimension if it's 1
                    elif yb.dim() == 2 and yb.shape[1] == 1:
                        yb = yb.squeeze(-1)  # Remove last dimension if it's 1
                loss = criterion(pred, yb)
                loss.backward()
                # Add gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(loss.item())
            
            current_loss = np.mean(epoch_losses)
            train_losses.append(current_loss)
            
            # Update learning rate scheduler
            scheduler.step(current_loss)
            
            # Log progress every 10 epochs
            if epoch % 10 == 0 or epoch <= 5:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch}/{self.epochs} – loss: {current_loss:.6f}, lr: {current_lr:.2e}")
            
            # Early stopping check with minimum delta
            if current_loss < best_loss - self.min_delta:
                best_loss = current_loss
                patience_counter = 0
                # Save best model state
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch} (best loss: {best_loss:.6f})")
                    # Restore best model state
                    self.model.load_state_dict(best_model_state)
                    break

        self.is_fitted = True
        self.training_metrics = {
            "train_mse": float(best_loss),
            "final_epoch": epoch,
            "train_losses": train_losses[-10:],  # Store last 10 losses
            "early_stopped": patience_counter >= self.patience
        }
        logger.info(f"[LSTM-Attention] Training completed after {epoch} epochs (best loss: {best_loss:.6f})")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.model is None or self.scaler is None:
            raise ValueError("Model not trained or scaler is missing")

        feats_df = self._prepare_features(df)
        feats_scaled = self.scaler.transform(feats_df.values)
        if len(feats_scaled) < self.context_length:
            raise ValueError("Insufficient length for prediction window")
        window = feats_scaled[-self.context_length :]
        x_tensor = torch.from_numpy(np.expand_dims(window, axis=0)).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_tensor).cpu().numpy().flatten()
        return pred  # returns prediction of shape (horizon,)

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        if "Close" not in df.columns or len(df) < self.context_length + self.horizon:
            return {}

        # The context for prediction is the dataframe excluding the values we want to predict
        context_df = df.iloc[: -self.horizon]
        preds = self.predict(context_df)

        # The actual values are the returns over the horizon
        close = df["Close"].values.astype(np.float32)
        last_price_in_context = close[-self.horizon - 1]
        actual_prices = close[-self.horizon :]

        if last_price_in_context > 1e-9:
            actual_returns = (actual_prices - last_price_in_context) / last_price_in_context
        else:
            actual_returns = np.zeros_like(actual_prices)

        mse = np.mean((preds - actual_returns) ** 2)
        return {"mse": float(mse), "rmse": float(np.sqrt(mse))}

    # ------------------------------------------------------------------
    # Serialization helpers (for consistency with pipeline)
    # ------------------------------------------------------------------
    def save_model(self, filepath: Union[str, Path]):
        state = {
            "model_state": self.model.state_dict() if self.model else None,
            "scaler": self.scaler,
            "config": {
                "context_length": self.context_length,
                "horizon": self.horizon,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "attention_heads": self.attention_heads,
            },
        }
        joblib.dump(state, filepath)
        logger.info(f"Saved LSTM-Attention model to {filepath}")

    def load_model(self, filepath: Union[str, Path]):
        state = joblib.load(filepath)
        self.scaler = state["scaler"]
        cfg = state["config"]
        for k, v in cfg.items():
            setattr(self, k, v)

        if not isinstance(self.scaler, StandardScaler) or not hasattr(self.scaler, "n_features_in_"):
            raise ValueError("Scaler is missing, not a StandardScaler, or is not fitted.")

        # Rebuild net
        self.model = _LSTMAttentionNet(
            input_dim=self.scaler.n_features_in_,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            attention_heads=self.attention_heads,
            horizon=self.horizon,
        )
        model_state = state.get("model_state")
        if model_state:
            self.model.load_state_dict(model_state)
        self.model.to(self.device)
        self.is_fitted = True
        logger.info(f"Loaded LSTM-Attention model from {filepath}")


def create_deep_learning_models(
    context_length: int = 60,
    horizon: int = 1,
    **kwargs,
) -> Dict[str, Any]:
    """Factory returning advanced DL models used in production trading."""
    models = {
        "lstm_attention": LSTMAttentionModel(context_length=context_length, horizon=horizon, **kwargs)
    }
    return models