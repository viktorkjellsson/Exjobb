import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
import numpy as np
import matplotlib.dates as mdates

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

def plot_reconstruction(dataset, model, N, feature_names, timestamps, ncol=2):
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
    fig, axes = plt.subplots(nrows=nrows, ncols=ncol, figsize=(10 * ncol, 4 * nrows))
    axes = axes.flatten()

    print(f'timestamps shape: {timestamps.shape}')

    for i, ax in enumerate(axes):
        if i < num_features:
            # Extract the data for plotting
            ax.plot(timestamps, data_subset[:, i], label="True", alpha=0.8)
            ax.plot(timestamps, reconstructed[:, i], label="Reconstructed", alpha=0.8)
            
            # Set title, labels, and grid
            ax.set_title(feature_names[i])
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid()

            # Set x-axis to show months
            ax.xaxis.set_major_locator(mdates.MonthLocator())  # Group by month
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format to 'Year-Month'

            # Set the tick positions first, then set the labels
            ax.set_xticks(ax.get_xticks())  # Get the current tick positions
            ax.set_xticklabels([mdates.DateFormatter('%Y-%m').format_data(t) for t in ax.get_xticks()], rotation=45)
        else:
            ax.axis("off")  # Hide unused subplots

    plt.tight_layout()
    plt.show()

def anomaly_score(original, reconstructed, timestamps, feature_names, ncol=2):
    """
    Calculate and plot anomaly scores for all features.

    Parameters:
    - original: Original data (numpy array, shape: [n_samples, n_features]).
    - reconstructed: Reconstructed data (numpy array, shape: [n_samples, n_features]).
    - timestamps: Timestamps for the x-axis (numpy array).
    - feature_names: List of feature names.
    - ncol: Number of columns in the subplot grid.
    """
    # Calculate anomaly scores as the absolute error between original and reconstructed
    anomaly_scores = np.abs(original - reconstructed)

    num_features = original.shape[1]  # Number of features
    feature_names = feature_names or [f"Feature {i+1}" for i in range(num_features)]

    # Determine subplot grid
    nrows = int(np.ceil(num_features / ncol))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncol, figsize=(10 * ncol, 4 * nrows))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_features:
            # Plot original, reconstructed, and anomaly scores for each feature
            ax.plot(timestamps, original[:, i], label="Original Data", alpha=0.5)
            ax.plot(timestamps, reconstructed[:, i], label="Reconstructed Data", alpha=0.5)
            ax.plot(timestamps, anomaly_scores[:, i], label="Anomaly Score", alpha=0.7, color='red')

            # Set title, labels, and grid
            ax.set_title(feature_names[i])
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid()

            # Set x-axis to show months
            ax.xaxis.set_major_locator(mdates.MonthLocator())  # Group by month
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format to 'Year-Month'

            # Set the tick positions first, then set the labels
            ax.set_xticks(ax.get_xticks())  # Get the current tick positions
            ax.set_xticklabels([mdates.DateFormatter('%Y-%m').format_data(t) for t in ax.get_xticks()], rotation=45)
        else:
            ax.axis("off")  # Hide unused subplots

    plt.tight_layout()
    plt.show()