import math
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

# ---------------------------------------------------------------------
# Helper dataset + positional encoding utils
# ---------------------------------------------------------------------

def _select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number])


class _SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class _PositionalEncoding(nn.Module):
    """Standard sine-cosine positional encoding."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        # Add positional encoding using the stored 'pe' buffer
        x = x + self.pe[:, :seq_len]
        return x


class _TransformerNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        horizon: int, # Horizon is kept for API consistency
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1) # Always output a single value

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        enc_out = self.encoder(x)
        last = enc_out[:, -1, :]  # Use representation of last time step
        out = self.fc(last)
        return out.squeeze(-1)


class TransformerModel:
    """Pure Transformer encoder model for financial time-series returns forecasting."""

    def __init__(
        self,
        context_length: int = 60,
        horizon: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = None,
    ):
        # Ensure d_model is divisible by nhead
        if d_model % nhead != 0:
            # Adjust d_model to be divisible by nhead
            d_model = ((d_model // nhead) + 1) * nhead
            logger.info(f"Adjusted d_model to {d_model} to be divisible by {nhead} attention heads")
        
        self.context_length = context_length
        self.horizon = horizon
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        if device is None or device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Optional[nn.Module] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted: bool = False
        self.training_metrics: Dict[str, Any] = {}

    # ------------------------- feature engineering -------------------
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = add_all_technical_indicators(df)
        df = add_microstructure_features(df)
        return _select_numeric(df).ffill().bfill()

    def _make_sequences(self, features: np.ndarray, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        seq_len, h = self.context_length, self.horizon
        for i in range(seq_len, len(features) - h):
            X.append(features[i - seq_len : i])
            y.append((close[i + h] - close[i]) / close[i])  # horizon return
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

    # ------------------------------ API ------------------------------
    def fit(self, df: pd.DataFrame) -> "TransformerModel":
        if "Close" not in df.columns:
            raise ValueError("Close column required")
        feats_df = self._prepare_features(df)
        self.scaler = StandardScaler()
        feats_scaled = self.scaler.fit_transform(feats_df.values)
        close_arr = df["Close"].values.astype(np.float32)
        X, y = self._make_sequences(feats_scaled, close_arr)
        if len(X) == 0:
            raise ValueError("Not enough data for sequences")
        dataset = _SeqDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.model = _TransformerNet(
            input_dim=X.shape[-1],
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            horizon=self.horizon,
            dropout=self.dropout,
        ).to(self.device)

        crit = nn.MSELoss()
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        patience = 10  # Early stopping patience
        
        for epoch in range(1, self.epochs + 1):
            epoch_losses = []
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = crit(pred, yb)
                loss.backward()
                opt.step()
                epoch_losses.append(loss.item())
            
            current_loss = np.mean(epoch_losses)
            logger.debug(f"Epoch {epoch}/{self.epochs} – loss {current_loss:.6f}")
            
            # Early stopping check
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch} (best loss: {best_loss:.6f})")
                    break
        self.training_metrics = {"train_mse": float(np.mean(epoch_losses))}
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.model is None or self.scaler is None:
            raise ValueError("Model not trained or components not available")
        feats_df = self._prepare_features(df)
        feats_scaled = self.scaler.transform(feats_df.values)
        if len(feats_scaled) < self.context_length:
            raise ValueError("Not enough data for prediction window")
        window = feats_scaled[-self.context_length :]
        x_tensor = torch.from_numpy(window[None, :, :]).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_tensor).cpu().numpy().flatten()
        return pred

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        preds = self.predict(df)
        close = df["Close"].values.astype(np.float32)
        actual = (close[-1] - close[-self.horizon - 1]) / close[-self.horizon - 1]
        mse = float((preds[0] - actual) ** 2)
        return {"mse": mse, "rmse": float(np.sqrt(mse))}

    # -------------------------- serialization ------------------------
    def save_model(self, filepath: Union[str, Path]):
        state = {
            "model_state": self.model.state_dict() if self.model else None,
            "scaler": self.scaler,
            "config": {
                "context_length": self.context_length,
                "horizon": self.horizon,
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
        }
        joblib.dump(state, filepath)
        logger.info(f"Saved Transformer model to {filepath}")

    def load_model(self, filepath: Union[str, Path]):
        state = joblib.load(filepath)
        if not isinstance(state, dict) or "scaler" not in state or "config" not in state or "model_state" not in state:
            raise ValueError("Invalid model state file")
        self.scaler = state["scaler"]
        cfg = state["config"]
        for k, v in cfg.items():
            setattr(self, k, v)
        
        # Ensure scaler is valid before accessing attributes
        if self.scaler is None or not hasattr(self.scaler, 'mean_'):
            raise ValueError("Scaler not loaded correctly or is invalid")

        self.model = _TransformerNet(
            input_dim=self.scaler.mean_.shape[0],
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            horizon=self.horizon,
            dropout=self.dropout,
        )
        self.model.load_state_dict(state["model_state"])
        self.model.to(self.device)
        self.is_fitted = True
        logger.info(f"Loaded Transformer model from {filepath}")


def create_transformer_models(context_length: int = 60, horizon: int = 1, **kwargs) -> Dict[str, Any]:
    return {
        "transformer": TransformerModel(context_length=context_length, horizon=horizon, **kwargs)
    }