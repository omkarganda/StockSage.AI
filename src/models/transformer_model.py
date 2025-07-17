"""
Transformer Model for StockSage.AI

This module implements a Transformer model for stock price prediction using
multi-head attention mechanisms.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime
import joblib

from ..utils.logging import get_logger

logger = get_logger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        pe = self.pe[:x.size(0), :]
        return x + pe


class TransformerNetwork(nn.Module):
    """Transformer Network for time series prediction"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_dim: int = 1
    ):
        super(TransformerNetwork, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        self.output_projection = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, d_model)
        
        # Apply transformer
        transformer_output = self.transformer_encoder(x)
        
        # Use the last time step for prediction
        output = transformer_output[:, -1, :]  # (batch_size, d_model)
        output = self.dropout(output)
        output = self.output_projection(output)  # (batch_size, output_dim)
        
        return output


class EarlyStopping:
    """Early stopping utility class"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop


class TransformerModel:
    """
    Transformer model for stock price prediction.
    
    This model uses transformer architecture with multi-head attention
    to capture complex temporal dependencies in financial time series.
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        device: str = 'auto'
    ):
        """
        Initialize Transformer Model.
        
        Parameters:
        -----------
        sequence_length : int, default=60
            Length of input sequences
        d_model : int, default=128
            Dimension of the model
        nhead : int, default=8
            Number of attention heads
        num_layers : int, default=3
            Number of transformer layers
        dim_feedforward : int, default=512
            Dimension of feedforward network
        dropout : float, default=0.1
            Dropout probability
        learning_rate : float, default=0.001
            Learning rate for optimization
        batch_size : int, default=32
            Batch size for training
        epochs : int, default=100
            Maximum number of training epochs
        early_stopping_patience : int, default=10
            Patience for early stopping
        device : str, default='auto'
            Device to use for computation ('cpu', 'cuda', or 'auto')
        """
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Model components
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False
        self.training_metrics = {}
        
        logger.info(f"Initialized TransformerModel with device: {self.device}")
    
    def _prepare_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for transformer training."""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(target[i + self.sequence_length])
            
        return np.array(X), np.array(y)
    
    def fit(self, df: pd.DataFrame, target_column: str = 'close') -> 'TransformerModel':
        """
        Train the Transformer model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data with features and target
        target_column : str, default='close'
            Name of the target column
            
        Returns:
        --------
        self : TransformerModel
            Fitted model
        """
        logger.info("Starting Transformer model training")
        
        try:
            # Prepare data
            feature_columns = [col for col in df.columns if col != target_column]
            X = df[feature_columns].values
            y = np.array(df[target_column].values)
            
            # Store feature names
            self.feature_names = feature_columns
            
            # Normalize features
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Prepare sequences
            X_seq, y_seq = self._prepare_sequences(X_scaled, y)
            
            # Split into train/validation
            split_idx = int(0.8 * len(X_seq))
            X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
            # Initialize model
            input_dim = X_train.shape[2]
            self.model = TransformerNetwork(
                input_dim=input_dim,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Early stopping
            early_stopping = EarlyStopping(patience=self.early_stopping_patience)
            
            # Training loop
            train_losses = []
            val_losses = []
            best_loss = float('inf')
            best_train_loss = float('inf')  # Track best training loss for metrics
            
            for epoch in range(self.epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                train_losses.append(train_loss)
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs.squeeze(), batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                
                # Track best losses
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_train_loss = train_loss  # Save corresponding training loss
                
                # Early stopping check
                if early_stopping(val_loss):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # FIXED: Use best training loss instead of final epoch loss
            self.training_metrics = {
                'train_mse': best_train_loss,  # Use best loss, not final epoch loss
                'val_mse': best_loss,
                'epochs_trained': epoch + 1,
                'early_stopped': early_stopping.early_stop
            }
            
            self.is_fitted = True
            logger.info(f"Training completed. Best train MSE: {best_train_loss:.6f}, Best val MSE: {best_loss:.6f}")
            
        except Exception as e:
            logger.error(f"Error during Transformer model training: {e}")
            raise
            
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data for prediction
            
        Returns:
        --------
        np.ndarray
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Prepare features
            X = df[self.feature_names].values
            if self.scaler is None:
                raise ValueError("Scaler is not fitted")
            X_scaled = self.scaler.transform(X)
            
            # Prepare sequences
            X_seq = []
            for i in range(len(X_scaled) - self.sequence_length + 1):
                X_seq.append(X_scaled[i:(i + self.sequence_length)])
            
            if len(X_seq) == 0:
                raise ValueError(f"Input data length {len(df)} is too short for sequence length {self.sequence_length}")
            
            X_seq = np.array(X_seq)
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            
            # Make predictions
            if self.model is None:
                raise ValueError("Model is not fitted")
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor)
                predictions = predictions.squeeze().cpu().numpy()
            
            # Handle single prediction
            if predictions.ndim == 0:
                predictions = np.array([predictions])
                
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def save(self, filepath: str) -> None:
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        if self.model is None:
            raise ValueError("Model is not fitted")
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'dim_feedforward': self.dim_feedforward,
            'training_metrics': self.training_metrics
        }
        
        torch.save(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> 'TransformerModel':
        """Load a trained model."""
        model_data = torch.load(filepath, map_location=self.device)
        
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.sequence_length = model_data['sequence_length']
        self.d_model = model_data['d_model']
        self.nhead = model_data['nhead']
        self.num_layers = model_data['num_layers']
        self.dim_feedforward = model_data['dim_feedforward']
        self.training_metrics = model_data['training_metrics']
        
        # Recreate model
        input_dim = len(self.feature_names)
        self.model = TransformerNetwork(
            input_dim=input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        ).to(self.device)
        
        self.model.load_state_dict(model_data['model_state_dict'])
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        return self
