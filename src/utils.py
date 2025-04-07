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


# def plot_lstm_results(train_data, prediction, timestamps):
#     train_data_np = train_data.numpy()
#     prediction_np = prediction.detach().numpy()

#     fig = go.Figure()

#     fig.add_trace(go.Scatter(x=timestamps, y=train_data_np, mode='lines', name='Original Data'))

#     fig.add_trace(go.Scatter(x=timestamps, y=prediction_np, mode='lines', name='Reconstructed Data'))

#     fig.update_layout(
#         title='LSTM Reconstruction vs. Original Data',
#         xaxis_title='Time',
#         yaxis_title='Strain',
#         template='plotly_white')

#     fig.show()

def calculate_anomalous_regions(original, reconstructed, mode):
    """
    Calculate anomalous regions based on the difference between original and reconstructed data.

    Parameters:
    - original: Original data (numpy array).
    - reconstructed: Reconstructed data (numpy array).
    - mode: Mode of anomaly detection (1 for absolute difference)

    Returns:
    - Anomalous regions as a boolean array.
    """

    if mode == 1:
        diff = np.abs(original - reconstructed)
        threshold = np.mean(diff) + 1 * np.std(diff)
        anomalous_regions = diff > threshold
    elif mode == 2:
        diff = np.abs(original - reconstructed)
        threshold = np.mean(diff) + 1 * np.std(diff)
        anomalous_regions = diff > threshold
        anomalous_regions = np.convolve(anomalous_regions.astype(int), np.ones(5), mode='same') >= 3
    
    return anomalous_regions, threshold

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

            anomalous_regions, threshold = calculate_anomalous_regions(data_subset[:, i], reconstructed[:, i], mode=2)
            anomaly_scores = np.abs(data_subset[:, i] - reconstructed[:, i])

            # threshold = np.percentile(anomaly_scores, 95)  # Example 
            # Definine threshold: 1) over percentile 2) over mean + 3*std, 3) fixed value
            # Define anomaly: 1) over threshold, 2) over threshold for a number of consequtive points, 3) mean of anomalies over a time window over threshold, 4) number of consequtive points over threshold x weight (weight = error / threshold)
            # anomalies_above_threshold = anomaly_scores > threshold
            # Extract the data for plotting
            ax.plot(timestamps, data_subset[:, i], label="True", alpha=0.8)
            ax.plot(timestamps, reconstructed[:, i], label="Reconstructed", alpha=0.8)
            ax.plot(timestamps, anomaly_scores, label="Anomaly Score", alpha=0.8)
            ax.axhline(y=threshold, color='red', linestyle='--', label='Threshold', alpha=0.6)

            ax.fill_between(timestamps, 0, anomaly_scores, where=anomalous_regions, color='red', alpha=0.3, label='Anomalous Region')
            
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

    return reconstructed

def anomaly_score(test, reconstructed, timestamps, feature_names, N, ncol=2):
    """
    Calculate and plot anomaly scores for all features.

    Parameters:
    - original: Original data (numpy array, shape: [n_samples, n_features]).
    - reconstructed: Reconstructed data (numpy array, shape: [n_samples, n_features]).
    - timestamps: Timestamps for the x-axis (numpy array).
    - feature_names: List of feature names.
    - ncol: Number of columns in the subplot grid.
    """

    num_features = reconstructed.shape[1]  # Number of features
    data_subset = test[:N]
    print(f'data_subset shape: {data_subset.shape}')
    print(f'reconstructed shape: {reconstructed.shape}')

    for i in range(num_features):

        # Calculate anomaly scores as the absolute error between original and reconstructed
        print(data_subset[:,i].shape)
        print(reconstructed[:,i].shape)
        anomaly_scores = np.abs(test[:,i] - reconstructed[:,i])


        feature_names = feature_names or [f"Feature {i+1}" for i in range(num_features)]

    # Determine subplot grid
    nrows = int(np.ceil(num_features / ncol))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncol, figsize=(10 * ncol, 4 * nrows))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_features:
            # Plot original, reconstructed, and anomaly scores for each feature
            ax.plot(timestamps, data_subset[:, i], label="Original Data", alpha=0.5)
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