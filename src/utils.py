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

def calculate_anomalous_regions(original, reconstructed, mode, k=1, n=18, error_threshold=0.1):
    """
    Calculate anomalous regions based on the difference between original and reconstructed data.

    Parameters:
    - original: Original data (numpy array).
    - reconstructed: Reconstructed data (numpy array).
    - mode: Mode of anomaly detection (1 for absolute difference, 2 for consecutive anomalies).
    - k: Threshold scaling factor (default is 1).
    - n: Minimum length for valid anomalous region (default is 5).
    - error_threshold: The minimum error value threshold for anomalies in mode 2 (default is 0.1).

    Returns:
    - Anomalous indices as a list of indices.
    - Threshold for anomaly detection.
    """
    error = np.abs(original - reconstructed)  # Calculate absolute error
    rolling_mean_error = np.convolve(error, np.ones((n,))/n, mode='valid')  # Rolling mean error

    threshold = np.mean(error) + k * np.std(error)  # Define the threshold for anomaly detection

    if mode == 1:
        anomalous_points = error > threshold
        anomalous_indices = np.where(anomalous_points)[0]
    
    if mode == 2 or mode == 3:
        anomalous_points = error > threshold  # Boolean array of anomalous points
        if mode == 3:
            anomalous_points = rolling_mean_error > threshold  # Boolean array of anomalous points

        anomalous_indices = np.where(anomalous_points)[0]
        # Group consecutive anomalous points into regions
        regions = []
        start_idx = anomalous_indices[0]  # Start with the first anomalous point
        
        for i in range(1, len(anomalous_indices)):
            # If we encounter a break in the consecutive sequence, close the previous region
            if anomalous_indices[i] != anomalous_indices[i - 1] + 1:
                if anomalous_indices[i - 1] - start_idx + 1 >= n:  # Only keep the region if it's long enough
                    regions.append((start_idx, anomalous_indices[i - 1]))
                start_idx = anomalous_indices[i]  # New region starts here
        
        # Add the last region if it's long enough
        if anomalous_indices[-1] - start_idx + 1 >= n:
            regions.append((start_idx, anomalous_indices[-1]))
        
        # Get the indices of valid regions
        valid_indices = []
        for start, end in regions:
            valid_indices.extend(range(start, end + 1))  # Add all indices in the valid regions
        
        anomalous_indices = valid_indices  # Only keep indices that are part of valid regions

    return anomalous_indices, threshold, error, rolling_mean_error

def plot_reconstruction(dataset, model, N, feature_names, timestamps, mode, ncol=1):
    """
    Plot true vs reconstructed values for every feature over N steps using subplots.

    Parameters:
    - dataset: Dataset or DataLoader containing the input data.
    - model: Trained autoencoder model.
    - N: Number of steps to plot.
    - feature_names: List of feature names (optional).
    - mode: Mode of anomaly detection (1 for absolute difference, 2 for consecutive anomalies).
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

    print(f"Shape of data_subset: {data_subset.shape}")
    print(f"Shape of reconstructed: {reconstructed.shape}")

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
    fig, axes = plt.subplots(nrows=nrows, ncols=ncol, figsize=(20 * ncol, 4 * nrows))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_features:
            # Calculate anomalies and threshold
            window_size = 18
            anomalous_indices, threshold, error, rolling_mean_error = calculate_anomalous_regions(data_subset[:, i], reconstructed[:, i], mode=mode)
            print(f'{feature_names[i]}\n  Anomalies: {anomalous_indices}\n  Threshold: {threshold}')
            

            # Plot the true data, reconstructed data, and error
            ax.plot(timestamps, data_subset[:, i], label="True", alpha=0.8)
            ax.plot(timestamps, reconstructed[:, i], label="Reconstructed", alpha=0.8)
            ax.plot(timestamps, error, label="Anomaly Score", alpha=0.8)
            
            # Plot the rolling mean error with a distinct label
            ax.plot(timestamps[window_size - 1:], rolling_mean_error, label="Rolling Mean Error", alpha=0.8, color='green')

            # Plot the threshold line
            ax.axhline(y=threshold, color='red', linestyle='--', label='Threshold', alpha=0.6)

            # Highlight anomalous regions
            anomalous_region_mask = np.zeros_like(error, dtype=bool)
            anomalous_region_mask[anomalous_indices] = True
            ax.fill_between(timestamps, 0, 1, where=anomalous_region_mask, color='red', alpha=0.3, label='Anomalous Region')

            # Set plot titles, labels, and grid
            ax.set_title(feature_names[i])
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid()

            # Format the x-axis to show months
            ax.xaxis.set_major_locator(mdates.MonthLocator())  # Group by month
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format to 'Year-Month'

            # Adjust x-tick labels for better readability
            ax.set_xticks(ax.get_xticks())  # Get current tick positions
            ax.set_xticklabels([mdates.DateFormatter('%Y-%m').format_data(t) for t in ax.get_xticks()], rotation=45)
        else:
            ax.axis("off")  # Hide unused subplots

    # Adjust layout and show the plot
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