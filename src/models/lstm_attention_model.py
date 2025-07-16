"""
LSTM Attention Model for StockSage.AI

This module implements an LSTM model with attention mechanism for stock price prediction.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime
import joblib

from ..utils.logging import get_logger

logger = get_logger(__name__)


class AttentionMechanism(nn.Module):
    """Attention mechanism for LSTM"""
    
    def __init__(self, hidden_dim: int):
        super(AttentionMechanism, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_outputs):
        # lstm_outputs: (batch_size, seq_len, hidden_dim)
        attention_weights = torch.softmax(self.attention(lstm_outputs), dim=1)
        # attention_weights: (batch_size, seq_len, 1)
        
        # Apply attention weights
        attended_output = torch.sum(attention_weights * lstm_outputs, dim=1)
        # attended_output: (batch_size, hidden_dim)
        
        return attended_output, attention_weights


class LSTMAttentionNetwork(nn.Module):
    """LSTM with Attention Network"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        super(LSTMAttentionNetwork, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = AttentionMechanism(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch_size, seq_len, hidden_dim)
        
        # Apply attention
        attended_out, attention_weights = self.attention(lstm_out)
        # attended_out: (batch_size, hidden_dim)
        
        # Apply dropout and final linear layer
        out = self.dropout(attended_out)
        out = self.fc(out)
        
        return out, attention_weights


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


class LSTMAttentionModel:
    """
    LSTM with Attention mechanism for stock price prediction.
    
    This model uses LSTM layers with attention mechanism to focus on the most
    relevant time steps for prediction.
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        device: str = 'auto'
    ):
        """
        Initialize LSTM Attention Model.
        
        Parameters:
        -----------
        sequence_length : int, default=60
            Length of input sequences
        hidden_dim : int, default=128
            Number of hidden units in LSTM layers
        num_layers : int, default=2
            Number of LSTM layers
        dropout : float, default=0.2
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
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
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
        
        logger.info(f"Initialized LSTMAttentionModel with device: {self.device}")
    
    def _prepare_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training."""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(target[i + self.sequence_length])
            
        return np.array(X), np.array(y)
    
    def fit(self, df: pd.DataFrame, target_column: str = 'close') -> 'LSTMAttentionModel':
        """
        Train the LSTM Attention model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data with features and target
        target_column : str, default='close'
            Name of the target column
            
        Returns:
        --------
        self : LSTMAttentionModel
            Fitted model
        """
        logger.info("Starting LSTM Attention model training")
        
        try:
            # Prepare data
            feature_columns = [col for col in df.columns if col != target_column]
            X = df[feature_columns].values
            y = df[target_column].values
            
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
            self.model = LSTMAttentionNetwork(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
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
                    outputs, _ = self.model(batch_X)
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
                        outputs, _ = self.model(batch_X)
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
            logger.error(f"Error during LSTM Attention model training: {e}")
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
            self.model.eval()
            with torch.no_grad():
                predictions, _ = self.model(X_tensor)
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
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'training_metrics': self.training_metrics
        }
        
        torch.save(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> 'LSTMAttentionModel':
        """Load a trained model."""
        model_data = torch.load(filepath, map_location=self.device)
        
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.sequence_length = model_data['sequence_length']
        self.hidden_dim = model_data['hidden_dim']
        self.num_layers = model_data['num_layers']
        self.training_metrics = model_data['training_metrics']
        
        # Recreate model
        input_dim = len(self.feature_names)
        self.model = LSTMAttentionNetwork(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.model.load_state_dict(model_data['model_state_dict'])
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        return self