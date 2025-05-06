"""
A PyTorch Dataset for loading, preprocessing, and batching time series strain sensor data from multiple CSV files.

This dataset performs the following:
- Loads CSV files from a given folder.
- Splits each file into training and testing data chronologically.
- Applies preprocessing and feature engineering on each split independently.
- Normalizes features using MinMaxScaler (fit on training data only).
- Concatenates features from all files (sensors) along the feature axis.
- Constructs sliding sequences of a specified length for model input.
- Returns sequences in PyTorch tensor format, ready for training.

Parameters
----------
folder_path : Path
    Path to the folder containing CSV files.

INPUT_FEATURES : list of str
    List of input feature names to extract and scale from each CSV file.

OUTPUT_FEATURES : list of str
    List of target feature names for potential downstream tasks.

sequence_length : int
    Length of each time series sequence (i.e., window size) for model input.

batch_size : int
    Number of sequences per batch in the PyTorch DataLoader.

test_size : float
    Proportion of each CSV file to allocate to the test set (e.g., 0.3 for 30%).

start_idx : int
    Number of initial rows to leave out from each CSV file if needed.

Attributes
----------
train_data : torch.Tensor
    Training data as a 3D tensor with shape [n_sequences, sequence_length, n_features].

test_data : torch.Tensor
    Testing data as a 3D tensor with shape [n_sequences, sequence_length, n_features].

train_dataloader : DataLoader
    PyTorch DataLoader for training sequences.

test_dataloader : DataLoader
    PyTorch DataLoader for testing sequences.

input_feature_names : list of str
    Input feature names prefixed by sensor (file) name, e.g., 'S-B_Close_Comp - Strain'.

output_feature_names : list of str
    Output feature names prefixed by sensor (file) name, e.g., 'S-B_Close_Comp - Temperature'.

timestamps_train : list
    List of datetime values for training samples (aligned with sequence start times).

timestamps_test : list
    List of datetime values for testing samples (aligned with sequence start times).

Methods
-------

"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# Add the root project directory to the Python path
ROOT = Path.cwd().parent.parent  # Adjust if needed
sys.path.append(str(ROOT))
from src.processing import preprocessing


class StrainDataset(Dataset):
    def __init__(self, folder_path, INPUT_FEATURES, OUTPUT_FEATURES, sequence_length, batch_size, test_size, start_idx):
        print("Initializing StrainDataset...")

        self.file_names = []  # Store file names
        train_data_list = []
        test_data_list = []
        train_timestamps = []
        test_timestamps = []

        # Initialize a dictionary to store a MinMaxScaler for each feature
        scalers = {feature: MinMaxScaler() for feature in INPUT_FEATURES}

        for file in folder_path.glob("*.csv"):
            print(f"\nProcessing file: {file.stem}")
            self.file_names.append(file.stem)
            df = pd.read_csv(file)

            # Initial trimming
            df = df.iloc[start_idx:-1].copy()
            df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

            # Split BEFORE feature engineering
            split_idx = int(len(df) * (1 - test_size))
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()

            # Save timestamps
            train_timestamps.extend(train_df["Time"].values)
            test_timestamps.extend(test_df["Time"].values)
            self.timestamps_train = train_timestamps
            self.timestamps_test = test_timestamps

            train_df = preprocessing.preprocessing_pipeline(train_df, interpolate_threshold=20)
            test_df = preprocessing.preprocessing_pipeline(test_df, interpolate_threshold=20)

            # Apply feature engineering independently
            train_df = preprocessing.add_features(train_df, column="Strain")
            test_df = preprocessing.add_features(test_df, column="Strain")

            # Extract features
            train_features = train_df[INPUT_FEATURES].fillna(0).to_numpy()
            test_features = test_df[INPUT_FEATURES].fillna(0).to_numpy()
            train_data_list.append(train_features)
            test_data_list.append(test_features)

        # Concatenate data from all sensors (horizontally)
        train_data = np.concatenate(train_data_list, axis=1)
        test_data = np.concatenate(test_data_list, axis=1)
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")

        # Feature name expansion
        expanded_feature_names = []
        for sensor_name in self.file_names:
            for f in INPUT_FEATURES:
                expanded_feature_names.append(f"{sensor_name} - {f}")
        self.feature_names = expanded_feature_names
        self.feature_count = train_data.shape[1]

        # # Store input and output feature names with sensor prefixes
        self.input_feature_names = []
        self.output_feature_names = []

        for sensor_name in self.file_names:
            self.input_feature_names.extend([f"{sensor_name} - {f}" for f in INPUT_FEATURES])
            self.output_feature_names.extend([f"{sensor_name} - {f}" for f in OUTPUT_FEATURES])

        self.input_feature_count = len(self.input_feature_names)
        self.output_feature_count = len(self.output_feature_names)

        train_scaled = train_data.copy()
        test_scaled = test_data.copy()
        print(f'Number of features: {len(INPUT_FEATURES)}')
        scaler = MinMaxScaler()
        for feature in INPUT_FEATURES:
            print(f'Feature: {feature}')
            feature_idx = [i for i, name in enumerate(self.input_feature_names) if feature in name]
            print(f"Feature indices {feature}: {feature_idx}")

            # Fit the scaler on the training data for each feature
            train_scaled[:, feature_idx] = scaler.fit_transform(train_data[:, feature_idx])
            test_scaled[:, feature_idx] = scaler.transform(test_data[:, feature_idx])
            print(f"Train data shape after scaling: {train_data.shape}, Test data shape after scaling: {test_data.shape}")

        # Generate sequences (rolling window approach)
        train_sequences = [train_scaled[i:i + sequence_length] for i in range(len(train_scaled) - sequence_length)]
        test_sequences = [test_scaled[i:i + sequence_length] for i in range(len(test_scaled) - sequence_length)]

        # Convert to NumPy arrays
        train_sequences_np = np.array(train_sequences)
        test_sequences_np = np.array(test_sequences)

        print(f"Train sequences shape: {train_sequences_np.shape}, Test sequences shape: {test_sequences_np.shape}")

        # Reshape back to [n_samples, sequence_length, n_features] if necessary
        self.train_data = torch.tensor(train_sequences_np, dtype=torch.float32)
        self.test_data = torch.tensor(test_sequences_np, dtype=torch.float32)

        # Store timestamps
        self.timestamps_train = train_timestamps[:len(train_sequences_np)]
        self.timestamps_test = test_timestamps[:len(test_sequences_np)]

        # DataLoaders
        self.train_dataloader = DataLoader(self.train_data, batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size, shuffle=False)

        # Overlap check via hashing
        train_hashes = [hash(tuple(seq.numpy().flatten())) for seq in self.train_data]
        test_hashes = [hash(tuple(seq.numpy().flatten())) for seq in self.test_data]
        overlap_hashes = set(train_hashes).intersection(set(test_hashes))
        print(f"Number of overlapping hashes: {len(overlap_hashes)}")
        if overlap_hashes:
            print("Overlapping sequences detected! Potential data leakage.")
        else:
            print("No overlapping sequences detected. Train-test split is clean.")