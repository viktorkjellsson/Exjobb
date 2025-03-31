import torch
from torch import nn, optim
from tqdm import tqdm 

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # print(f"Input to LSTM (x) shape: {x.shape}")
        lstm_out, _ = self.lstm(x)
        # print(f"Output from LSTM (lstm_out) shape: {lstm_out.shape}")
        prediction = self.fc(lstm_out[:, -1, :])
        # print(f"Output from Fully Connected Layer (prediction) shape: {prediction.shape}")
        return prediction
    
def training_loop(model, train_loader, num_epochs, learning_rate, print_every=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        # Wrap `train_loader` with tqdm to add a progress bar
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")):
            # Uncomment if needed: batch = batch.unsqueeze(-1).float()
            optimizer.zero_grad()
            prediction = model(batch)
            loss = criterion(prediction, batch[:, -1, :])
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Optionally print batch loss (every `print_every` batches)
            # if (batch_idx + 1) % print_every == 0:
                # tqdm.write(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        losses.append(epoch_loss/len(train_loader))

        print(f"\nEpoch {epoch+1}/{num_epochs}, Average Loss: {losses[epoch]:.4f}\n")

    return losses

def plot_train()
    
# def training_loop(model, train_loader, num_epochs, learning_rate, print_every=10):
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     for epoch in range(num_epochs):
#         model.train()
#         epoch_loss = 0

#         for batch in train_loader:
#             # print(batch.shape)
#             # batch = batch.unsqueeze(-1).float()
#             optimizer.zero_grad()
#             prediction = model(batch)
#             loss = criterion(prediction, batch[:, -1, :])
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()

#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}\n')