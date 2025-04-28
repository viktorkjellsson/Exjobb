import torch
from torch import nn, optim
from tqdm import tqdm 
import sys
from pathlib import Path

# Add the root project directory to the Python path
ROOT = Path.cwd().parent  # This will get the project root since the notebook is in 'notebooks/'
sys.path.append(str(ROOT))
from configs.path_config import LOGS_DIR
from src.train_logger import TrainingLogger


class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout):    
        super(LSTMModel, self).__init__()

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
    model_folder, 
    model_name, 
    input_features, 
    output_features, 
    input_feature_names,
    output_feature_names 
):
    
    # Get feature indices once
    input_indices = [input_feature_names.index(f) for f in input_feature_names]
    output_indices = [input_feature_names.index(f) for f in output_feature_names]
    # input_indices = list(range(20))  # Select first 20 features
    # output_indices = list(range(0, 19, 4))
    print(f"Input indices: {input_indices}, Output indices: {output_indices}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Initialize Logger
    logger = TrainingLogger(log_dir=LOGS_DIR)
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
    )
    logger.start_timer()

    losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        # Wrap train_loader with tqdm to add a progress bar
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")):
            # Uncomment if needed: batch = batch.unsqueeze(-1).float()
            # x = batch[:, :, input_indices].to(device)
            x = batch.to(device)         # model input
            y = batch[:, -1, output_indices].to(device)         # reconstruction target at last timestep
            optimizer.zero_grad()
            prediction = model(x)
            # loss = criterion(prediction, y)
            loss = criterion(prediction[:, output_indices], y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        logger.log_epoch_loss(epoch, avg_loss)
        print(f"\nEpoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}\n")
    
    logger.end_timer()
    logger.save_log()
    
    savepath =  model_folder / model_name
    torch.save(model, savepath)
    
    return losses, prediction