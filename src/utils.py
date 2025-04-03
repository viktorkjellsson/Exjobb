import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
import numpy as np

def plot_epochs_loss(num_epochs, losses):

    plt.plot(range(1, num_epochs + 1), losses)
    plt.xlabel('Epochs')
    plt.xticks(range(1, num_epochs + 1))
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.show()


def plot_lstm_results(train_data, prediction, timestamps):
    train_data_np = train_data.numpy()
    prediction_np = prediction.detach().numpy()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=timestamps, y=train_data_np, mode='lines', name='Original Data'))

    fig.add_trace(go.Scatter(x=timestamps, y=prediction_np, mode='lines', name='Reconstructed Data'))

    fig.update_layout(
        title='LSTM Reconstruction vs. Original Data',
        xaxis_title='Time',
        yaxis_title='Strain',
        template='plotly_white')

    fig.show()

def plot_reconstruction(dataset, model, N, feature_names=None, ncol=3):
    """
    Plot true vs reconstructed values for every feature over N steps using subplots.

    Parameters:
    - dataset: Dataset or DataLoader containing the input data.
    - model: Trained autoencoder model.
    - N: Number of steps to plot.
    - feature_names: List of feature names (optional).
    - ncol: Number of columns in the subplot grid.
    """
    model.eval()  # Set model to evaluation mode
    device = next(model.parameters()).device  # Get model's device

    # Select first N samples
    if isinstance(dataset, torch.Tensor):
        data_subset = dataset[:N]
    else:
        data_subset = torch.stack([dataset[i] for i in range(N)])

    data_subset = data_subset.to(device)

    with torch.no_grad():
        reconstructed = model(data_subset)

    # Convert to CPU
    data_subset = data_subset.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()

    # Extract last timestep if sequences are present
    if len(data_subset.shape) == 3:  # (N, seq_len, num_features)
        data_subset = data_subset[:, -1, :]  # Extract last time step

    if len(reconstructed.shape) == 3:  # Ensure reconstructed matches shape
        reconstructed = reconstructed[:, -1, :]

    num_features = data_subset.shape[1]  # Number of features

    # Ensure feature names are valid
    feature_names = feature_names or [f"Feature {i+1}" for i in range(num_features)]

    # Determine subplot grid
    nrows = int(np.ceil(num_features / ncol))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncol, figsize=(5 * ncol, 4 * nrows))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_features:
            ax.plot(range(N), data_subset[:, i], label="True", alpha=0.8)
            ax.plot(range(N), reconstructed[:, i], label="Reconstructed", alpha=0.8)
            ax.set_title(feature_names[i])
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid()
        else:
            ax.axis("off")  # Hide unused subplots

    plt.tight_layout()
    plt.show()