"""
LSTM Training Script for Multivariate Time Series Prediction.

This module defines a PyTorch-based LSTM model and a training loop tailored for
multivariate time series prediction tasks.
It includes training utilities like device management, logging, and threshold estimation
for downstream use.

Classes:
--------
- LSTMModel: LSTM-based regressor with dropout and linear output.

Functions:
----------
- training_loop(...): Trains the LSTM model using a specified DataLoader, logs performance metrics,
  saves the model and error thresholds.

Dependencies:
-------------
- torch: For model definition and training.
- tqdm: For progress bars during training.
- numpy: For statistical computations.
- json: For saving training thresholds.
- pathlib.Path: For path operations.
- TrainingLogger: Custom logger for saving model parameters and training time.

Usage:
------
This script is typically used within a larger ML pipeline. To train a model, instantiate
`LSTMModel`, prepare a DataLoader, and call `training_loop(...)` with appropriate arguments.

Example:

model = LSTMModel(input_dim=10, output_dim=2, hidden_dim=128, num_layers=2, dropout=0.3)
losses, preds = training_loop(
    model=model,
    train_loader=dataloader,
    num_epochs=10,
    learning_rate=0.001,
    models_folder='outputs/models/',
    model_subname='strain_predictor',
    input_features=['strain', 'temperature',
    output_features=['strain,
    input_feature_names=['strain1', 'temperature1 ..., 'strain10', 'temperature10'],
    output_feature_names=['strain9', 'strain10']
)
"""

import torch
from torch import nn, optim
from tqdm import tqdm 
import sys
from pathlib import Path
import numpy as np
import json

# Add the root project directory to the Python path
ROOT = Path.cwd().parent  # This will get the project root since the notebook is in 'notebooks/'
sys.path.append(str(ROOT))
from configs.path_config import LOGS_DIR
from src.train_logger import TrainingLogger


class LSTMModel(nn.Module):
    # def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout):    
        super(LSTMModel, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)                        # [batch, seq_len, hidden]
        out = self.dropout(lstm_out[:, -1, :])            # final timestep output
        prediction = self.fc(out)                         # [batch, input_dim]
        return prediction
    
def training_loop(
    model, 
    train_loader, 
    num_epochs, 
    learning_rate, 
    models_folder, 
    model_subname, 
    input_features, 
    output_features, 
    input_feature_names,
    output_feature_names 
):
    
    # Get feature indices once
    input_indices = [input_feature_names.index(f) for f in input_feature_names]
    output_indices = [input_feature_names.index(f) for f in output_feature_names]
    print(f"Input indices: {input_indices}, Output indices: {output_indices}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    model_folder = f'lstm_model_{model_subname}_{model.input_dim}_{model.hidden_dim}_{model.num_layers}_{learning_rate}_{model.dropout_rate}'
    model_folder_path = Path(models_folder) / model_folder
    model_folder_path.mkdir(parents=True, exist_ok=True)  # Ensure it exists
    # model_name = model_folder_path / model_subname / 'model.pth'  # Safe path concatenation

    # Initialize Logger
    logger = TrainingLogger(log_dir=model_folder_path)
    logger.log_parameters(
        input_dim=model.lstm.input_size,
        output_dim=model.fc.out_features,
        hidden_dim=model.lstm.hidden_size,
        num_layers=model.lstm.num_layers,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        dropout=model.dropout.p,
        input_features=input_features,
        output_features=output_features,
        input_feature_names=input_feature_names,
        output_feature_names=output_feature_names,
    )
    logger.start_timer()

    losses = []
    epoch_losses = []
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")):
            x = batch.to(device)  # model input
            y = batch[:, -1, output_indices].to(device)  # reconstruction target at last timestep
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction[:, output_indices], y)
            loss.backward()
            optimizer.step()

            # Append batch loss to epoch_losses
            epoch_losses.append(loss.item())

        # Calculate the average loss using np.mean
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        logger.log_epoch_loss(epoch, avg_loss)
        print(f"\nEpoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}\n")


    # After training, calculate the mean and standard deviation of errors
    mean_error = avg_loss  # Mean of all per-batch losses
    std_error = np.std(epoch_losses)    # Standard deviation of all per-batch losses

    print(f"Mean error: {mean_error:.4f}\nStandard deviation error: {std_error:.4f}")

    threshold_path = model_folder_path / 'threshold.json'
    with open(threshold_path, 'w') as f:
        json.dump({'mean_error': float(mean_error), 'std_error': float(std_error)}, f)
    
    # Save the model
    model_path = model_folder_path / 'model.pth'
    torch.save(model, model_path)

    logger.end_timer()
    logger.save_log()
    
    return losses, prediction