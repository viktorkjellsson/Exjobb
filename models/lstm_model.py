import torch
from torch import nn, optim
from tqdm import tqdm 
import sys
from pathlib import Path

# Add the root project directory to the Python path
ROOT = Path.cwd().parent  # This will get the project root since the notebook is in 'notebooks/'
sys.path.append(str(ROOT))
from configs.path_config import OUTPUT_DIR, WEIGHTS_DIR
from src.train_logger import TrainingLogger


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        prediction = self.fc(lstm_out[:, -1, :])
        return prediction
    
def training_loop(model, train_loader, num_epochs, learning_rate, log_dir=OUTPUT_DIR/'logs'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize Logger
    logger = TrainingLogger(log_dir=log_dir)
    logger.log_parameters(
        input_dim=model.lstm.input_size,
        hidden_dim=model.lstm.hidden_size,
        num_layers=model.lstm.num_layers,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )
    logger.start_timer()

    losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        # Wrap train_loader with tqdm to add a progress bar
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")):
            # Uncomment if needed: batch = batch.unsqueeze(-1).float()
            optimizer.zero_grad()
            prediction = model(batch)
            loss = criterion(prediction, batch[:, -1, :])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        logger.log_epoch_loss(epoch, avg_loss)
        print(f"\nEpoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}\n")
    
    logger.end_timer()
    logger.save_log()

    savepath =  WEIGHTS_DIR / 'weights.pth'
    torch.save(model, savepath)
    
    return losses
    