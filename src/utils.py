import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
import numpy as np
import matplotlib.dates as mdates
import pandas as pd
import json
from itertools import groupby
from operator import itemgetter
from plotly.subplots import make_subplots

def load_model_and_threshold(model_path):
    model = torch.load(model_path / 'model.pth')
    model.eval()

    with open(model_path / 'threshold.json', 'r') as f:
        threshold_data = json.load(f)
    
    mean_error = threshold_data.get('mean_error', None)  # Extract the mean error value from training
    std_error = threshold_data.get('std_error', None)  # Extract the standard deviation error value from training

    return model, mean_error, std_error

def plot_epochs_loss(num_epochs, losses):

    plt.figure(figsize=(15, 9))
    plt.plot(range(1, num_epochs + 1), losses)
    plt.xlabel('Epochs')
    plt.xticks(range(1, num_epochs + 1))
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.show()


def sort_anomalies(anomalous_indices, timestamps):
    
    """
    Find regions with more than 'threshold' consecutive NaNs in the 'Strain' column of the DataFrame.

    Args: 
        df (pd.DataFrame): The input DataFrame.
        threshold (int): Minimum number of consecutive NaN values to consider as a region.

    Returns:
        consecutive_nan_regions (list): List of tuples containing start and end indices of consecutive NaN regions.
        nan_regions_sorted (list): Sorted list including start & end times, indices, and length.
    """
    print(f"\n[DEBUG] anomalous_indices: {anomalous_indices}")
    consecutive_anomalies = []
    start_idx = None

    for i in range(len(anomalous_indices)):
        if start_idx is None:
            start_idx = anomalous_indices[i]
        if i == len(anomalous_indices) - 1 or anomalous_indices[i] + 1 != anomalous_indices[i + 1]:
            end_idx = anomalous_indices[i]
            if (end_idx - start_idx + 1) >= 1:
                consecutive_anomalies.append((start_idx, end_idx))
            start_idx = None

    print(f"[DEBUG] consecutive_anomalies: {consecutive_anomalies}")
    print(f"[DEBUG] timestamps length: {len(timestamps)}")

    if not consecutive_anomalies:
        print("[DEBUG] No consecutive anomalies found.")
        return pd.DataFrame()

    data = [{
        'Start': timestamps[start],
        'End': timestamps[end],
        'Length': f"{(timestamps[end] - timestamps[start]).days} days, {(timestamps[end] - timestamps[start]).seconds // 3600} hours"
    } for start, end in consecutive_anomalies]

    df_anomalies = pd.DataFrame(data)
    print(f"[DEBUG] Anomalies DataFrame:\n{df_anomalies}")
    df_anomalies.sort_values(by='Start', ascending=True, inplace=True)
    df_anomalies.reset_index(drop=True, inplace=True)

    return df_anomalies


def calculate_anomalous_regions(original, reconstructed, threshold, mode, k=1, n=18):
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

        if len(anomalous_indices) == 0:
            print("No anomalies detected.")
            return [], threshold, error, rolling_mean_error
        
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

#### WITH SUBPLOTS ####
# def plot_reconstruction(dataset, model, N, input_feature_names, output_feature_names, timestamps, threshold, save_dir, mode):
#     model.eval()
#     device = next(model.parameters()).device

#     # Prepare the data subset
#     if isinstance(dataset, torch.Tensor):
#         data_subset = dataset[:N]
#     else:
#         data_subset = torch.stack([dataset[i] for i in range(N)])
#     data_subset = data_subset.to(device)

#     with torch.no_grad():
#         reconstructed = model(data_subset)

#     # Move to CPU for plotting
#     data_subset = data_subset.cpu().numpy()
#     reconstructed = reconstructed.cpu().numpy()

#     # Handle sequence data (take last time step)
#     if len(data_subset.shape) == 3:
#         data_subset = data_subset[:, -1, :]
#     if len(reconstructed.shape) == 3:
#         reconstructed = reconstructed[:, -1, :]

#     output_indices = [input_feature_names.index(f) for f in output_feature_names]
#     num_features = len(output_indices)

#     # Create subplots for each feature
#     subplot_titles = [f"{input_feature_names[feature_idx]}" for feature_idx in output_indices]
#     fig = make_subplots(rows=num_features, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=subplot_titles)

#     # Loop over features to add traces
#     for i, feature_idx in enumerate(output_indices):
#         feature_name = input_feature_names[feature_idx]

#         # Anomaly calculation
#         window_size = 18
#         anomalous_indices, threshold, error, rolling_mean_error = calculate_anomalous_regions(
#             data_subset[:, feature_idx], reconstructed[:, feature_idx], threshold, mode=mode
#         )
#         df_anomalies = sort_anomalies(anomalous_indices, timestamps)

#         # Plot true data (only show legend for the first feature)
#         fig.add_trace(go.Scatter(x=timestamps, y=data_subset[:, feature_idx],
#                                 mode='lines', name='True', line=dict(color='#1f77b4', width=1),
#                                 showlegend=False if i > 0 else True),
#                       row=i+1, col=1)

#         # Plot reconstructed data (only show legend for the first feature)
#         fig.add_trace(go.Scatter(x=timestamps, y=reconstructed[:, feature_idx],
#                                 mode='lines', name='Reconstructed', line=dict(color='#ff7f0e', width=1),
#                                 showlegend=False if i > 0 else True),
#                       row=i+1, col=1)

#         # Plot anomaly score (only show legend for the first feature)
#         fig.add_trace(go.Scatter(x=timestamps, y=error,
#                                 mode='lines', name='Anomaly Score', line=dict(color='#98df8a'),
#                                 showlegend=False if i > 0 else True),
#                       row=i+1, col=1)

#         # Plot rolling mean error (only show legend for the first feature)
#         fig.add_trace(go.Scatter(x=timestamps[window_size - 1:], y=rolling_mean_error,
#                                 mode='lines', name='Rolling Mean Error',
#                                 line=dict(color='#2ca02c'), showlegend=False if i > 0 else True),
#                       row=i+1, col=1)

#         # Plot threshold line (only show legend for the first feature)
#         fig.add_trace(go.Scatter(x=[timestamps[0], timestamps[-1]], y=[threshold, threshold],
#                                 mode='lines', name=f'Threshold',
#                                 line=dict(color='#d62728', dash='dash'),
#                                 showlegend=False if i > 0 else True),
#                       row=i+1, col=1)

#         if anomalous_indices:
#             for k, g in groupby(enumerate(sorted(anomalous_indices)), lambda ix: ix[0] - ix[1]):
#                 group = list(map(itemgetter(1), g))
#                 start_idx, end_idx = group[0], group[-1]

#                 fig.add_trace(go.Scatter(
#                     x=timestamps[start_idx:end_idx+1],
#                     y=data_subset[start_idx:end_idx+1, feature_idx],
#                     mode='lines',
#                     name='Anomalous Region',
#                     fill='tozeroy',
#                     fillcolor='rgba(255, 152, 150, 0.6)',  # 20% opacity
#                     line=dict(color='rgba(255, 152, 150, 0.0)'),  # Invisible line
#                     showlegend=False
#                 ), row=i+1, col=1)

#         # Add y-axis title for each subplot
#         fig.update_yaxes(title_text='Strain (normalized)', row=i+1, col=1)
#         if i == num_features - 1:  # Add x-axis title only for the last subplot
#             fig.update_xaxes(title_text="Time", row=i+1, col=1)
#         else:
#             fig.update_xaxes(title_text="", row=i+1, col=1)  # Remove title for others

#     # Update layout for all subplots (common settings)
#     fig.update_layout(
#         xaxis_title="Time",
#         legend=dict(
#             orientation='h',
#             x=1,  # Right-align the legend
#             y=1,  # Position it at the top
#             xanchor='right',  # Anchor to the right
#             yanchor='top'  # Anchor to the top
#         ),
#         template="plotly_white",
#         margin=dict(r=100, t=100),  # Adjust margins to prevent clipping
#         height=400 * num_features  # Adjust the height based on the number of subplots
#     )

#     # Set x-axis ticks for all subplots
#     fig.update_xaxes(
#         tickangle=45,
#         dtick="M1",  # One tick per month
#         tickformat="%b %Y",  # e.g., Jan 2025
#         ticklabelmode="period",  # Align labels to whole months
#         showticklabels=True  # Show tick labels on all subplots
#     )

#     # Ensure the save directory exists
#     save_dir.parent.mkdir(parents=True, exist_ok=True)

#     # Define width and height to preserve aspect ratio
#     width = 1600  # Customize as needed
#     height = 400 * num_features  # Adjust based on the number of features/subplots

#     # Save the plot as a PDF with the specified dimensions
#     fig.write_image(str(save_dir), format='pdf', width=width, height=height, scale=1)

#     # Show the plot
#     fig.show()

#     return reconstructed

### WITHOUT SUBPLOTS ####
def plot_reconstruction(dataset, model, N, input_feature_names, output_feature_names, timestamps, threshold, save_dir, mode):
    model.eval()
    device = next(model.parameters()).device

    # Prepare the data subset
    if isinstance(dataset, torch.Tensor):
        data_subset = dataset[:N]
    else:
        data_subset = torch.stack([dataset[i] for i in range(N)])
    data_subset = data_subset.to(device)

    with torch.no_grad():
        reconstructed = model(data_subset)

    # Move to CPU for plotting
    data_subset = data_subset.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()

    # Handle sequence data (take last time step)
    if len(data_subset.shape) == 3:
        data_subset = data_subset[:, -1, :]
    if len(reconstructed.shape) == 3:
        reconstructed = reconstructed[:, -1, :]

    output_indices = [input_feature_names.index(f) for f in output_feature_names]

    for feature_idx in output_indices:
        feature_name = input_feature_names[feature_idx]

        # Anomaly detection
        window_size = 18
        anomalous_indices, thres, error, rolling_mean_error = calculate_anomalous_regions(
            data_subset[:, feature_idx], reconstructed[:, feature_idx], threshold, mode=mode
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=timestamps, y=data_subset[:, feature_idx],
                                 mode='lines', name='True', line=dict(color='#1f77b4', width=1)))
        fig.add_trace(go.Scatter(x=timestamps, y=reconstructed[:, feature_idx],
                                 mode='lines', name='Reconstructed', line=dict(color='#ff7f0e', width=1)))
        fig.add_trace(go.Scatter(x=timestamps, y=error,
                                 mode='lines', name='Anomaly Score', line=dict(color='#98df8a')))
        fig.add_trace(go.Scatter(x=timestamps[window_size - 1:], y=rolling_mean_error,
                                 mode='lines', name='Rolling Mean Error', line=dict(color='#2ca02c')))
        fig.add_trace(go.Scatter(x=[timestamps[0], timestamps[-1]], y=[thres, thres],
                                 mode='lines', name='Threshold', line=dict(color='#d62728', dash='dash')))

        if anomalous_indices:
            for k, g in groupby(enumerate(sorted(anomalous_indices)), lambda ix: ix[0] - ix[1]):
                group = list(map(itemgetter(1), g))
                start_idx, end_idx = group[0], group[-1]
                fig.add_trace(go.Scatter(
                    x=timestamps[start_idx:end_idx + 1],
                    y=data_subset[start_idx:end_idx + 1, feature_idx],
                    mode='lines',
                    name='Anomalous Region',
                    fill='tozeroy',
                    fillcolor='rgba(255, 152, 150, 0.6)',
                    line=dict(color='rgba(255, 152, 150, 0.0)'),
                    showlegend=False
                ))

        fig.update_layout(
            title=f"Reconstruction - {feature_name}",
            xaxis_title="Time",
            yaxis_title="Strain (normalized)",
            legend=dict(orientation='h', x=1, y=1, xanchor='right', yanchor='top'),
            template="plotly_white",
            margin=dict(r=100, t=100),
            width=1600,
            height=500
        )

        fig.update_xaxes(
            tickangle=45,
            dtick="M1",
            tickformat="%b %Y",
            ticklabelmode="period"
        )

        # Save to separate file
        file_path = save_dir.parent / f"{save_dir.stem}_{feature_name}{save_dir.suffix}"
        fig.write_image(str(file_path), format='pdf', width=1600, height=500, scale=1)

        fig.show()



### WITH MATPLOTLIB ####
# def plot_reconstruction(dataset, model, N, input_feature_names, output_feature_names, timestamps, threshold, mode, ncol=1):
#     """
#     Plot true vs reconstructed values for every feature over N steps using subplots.

#     Parameters:
#     - dataset: Dataset or DataLoader containing the input data.
#     - model: Trained autoencoder model.
#     - N: Number of steps to plot.
#     - output_feature_names: List of feature names (optional).
#     - mode: Mode of anomaly detection (1 for absolute difference, 2 for consecutive anomalies).
#     - ncol: Number of columns in the subplot grid.
#     """
#     model.eval()  # Set model to evaluation mode
#     device = next(model.parameters()).device  # Get model's device

#     # Select first N samples
#     if isinstance(dataset, torch.Tensor):
#         data_subset = dataset[:N]
#     else:
#         data_subset = torch.stack([dataset[i] for i in range(N)])

#     data_subset = data_subset.to(device)

#     with torch.no_grad():
#         reconstructed = model(data_subset)

#     # Convert to CPU
#     data_subset = data_subset.cpu().numpy()
#     reconstructed = reconstructed.cpu().numpy()

#     print(f"Shape of data_subset: {data_subset.shape}")
#     print(f"Shape of reconstructed: {reconstructed.shape}")

#     # Extract last timestep if sequences are present
#     if len(data_subset.shape) == 3:  # (N, seq_len, num_features)
#         data_subset = data_subset[:, -1, :]  # Extract last time step

#     if len(reconstructed.shape) == 3:  # Ensure reconstructed matches shape
#         reconstructed = reconstructed[:, -1, :]

#     # # Ensure feature names are valid
#     # feature_names = feature_names or [f"Feature {i+1}" for i in range(num_features)]
#     # print("Features----------------------------------\n" + "\n".join(feature_names))

#     output_indices = [input_feature_names.index(f) for f in output_feature_names]
#     print(f"Output indices: {output_indices}")

#     num_features = len(output_indices)
#     print(f"Number of features: {num_features}")
#     print(f"Output feature names: {output_feature_names}")

#     # Determine subplot grid
#     nrows = int(np.ceil(num_features / ncol))
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncol, figsize=(20 * ncol, 4 * nrows))
#     # axes = axes.flatten()

#     if len(output_indices) == 1:
#         axes = [axes]  # Make it a list to handle it like a loop
#     plt_idx = 0
#     for i, ax in zip(output_indices, axes):
#         # if i < num_features:
#         feature_name = input_feature_names[i]
#         print(f"Feature Name: {feature_name}")
#         print(f"Feature Index: {i}")

#         # Calculate anomalies and threshold
#         window_size = 18
#         anomalous_indices, threshold, error, rolling_mean_error = calculate_anomalous_regions(data_subset[:, i], reconstructed[:, i], threshold, mode=mode)
#         df_anomalies = sort_anomalies(anomalous_indices, timestamps)
#         print(f'\n{feature_name} Anomalies:\n{df_anomalies}')
        

#         # Plot the true data, reconstructed data, and error
#         ax.plot(timestamps, data_subset[:, i], label="True", alpha=0.8)
#         ax.plot(timestamps, reconstructed[:, i], label="Reconstructed", alpha=0.8)
#         ax.plot(timestamps, error, label="Anomaly Score", alpha=0.8)
#         print(f'plt_idx: {plt_idx}, i: {i}')
#         plt_idx += 4
        
#         # Plot the rolling mean error with a distinct label
#         ax.plot(timestamps[window_size - 1:], rolling_mean_error, label="Rolling Mean Error", alpha=0.8, color='green')

#         # Plot the threshold line
#         ax.axhline(y=threshold, color='red', linestyle='--', label='Threshold', alpha=0.6)

#         # Highlight anomalous regions
#         anomalous_region_mask = np.zeros_like(error, dtype=bool)
#         anomalous_region_mask[anomalous_indices] = True
#         ax.fill_between(timestamps, 0, 1, where=anomalous_region_mask, color='red', alpha=0.3, label='Anomalous Region')

#         # Set plot titles, labels, and grid
#         ax.set_title(feature_name)
#         ax.set_xlabel("Time")
#         ax.set_ylabel("Value")
#         ax.legend()
#         ax.grid()

#         # Format the x-axis to show months
#         ax.xaxis.set_major_locator(mdates.MonthLocator())  # Group by month
#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format to 'Year-Month'

#         # Adjust x-tick labels for better readability
#         ax.set_xticks(ax.get_xticks())  # Get current tick positions
#         ax.set_xticklabels([mdates.DateFormatter('%Y-%m').format_data(t) for t in ax.get_xticks()], rotation=45)

#     # Adjust layout and show the plot
#     plt.tight_layout()
#     plt.show()

#     return reconstructed


def anomaly_score(test, reconstructed, timestamps, output_feature_names, N, ncol=2):
    """
    Calculate and plot anomaly scores for all features.

    Parameters:
    - original: Original data (numpy array, shape: [n_samples, n_features]).
    - reconstructed: Reconstructed data (numpy array, shape: [n_samples, n_features]).
    - timestamps: Timestamps for the x-axis (numpy array).
    - output_feature_names: List of feature names.
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


        output_feature_names = output_feature_names or [f"Feature {i+1}" for i in range(num_features)]

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