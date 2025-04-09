from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np
import sys
from torch.utils.data import Dataset, DataLoader
import torch

# Add the root project directory to the Python path
ROOT = Path.cwd().parent.parent  # This will get the project root since the notebook is in 'notebooks/'
sys.path.append(str(ROOT))
from src.processing import preprocessing

class StrainDataset(Dataset):
    def __init__(self, folder_path, features, sequence_length, start_idx, test_size):
        self.sequences = []
        self.timestamps = []
        self.timestamps_train = []
        self.timestamps_test = []  
        self.file_names = []  # Store file names without .csv

        multivariate_data = []

        # Load and process each .csv file in the folder
        for file in folder_path.glob("*.csv"):
            self.file_names.append(file.stem)
            df = pd.read_csv(file)
            df = df.iloc[start_idx:-1]  # Trim data based on start index
            
            df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
            self.timestamps.append(df["Time"].values)  # Store timestamps
            
            # Apply preprocessing pipeline
            processed_df = preprocessing.preprocessing_pipeline(df, interpolate_threshold=60)  # Process the data
            
            # Apply feature engineering
            processed_df = preprocessing.add_features(processed_df, column="Strain", window=12)
            print(processed_df.columns)  # Debugging line to check columns

            strain_series = processed_df[features].fillna(0).to_numpy()
            multivariate_data.append(strain_series)  # Append the data from each file
        
        # After the loop, concatenate all the data
        multivariate_data = np.concatenate(multivariate_data, axis=1)  # Concatenate along the correct axis
        self.feature_count = multivariate_data.shape[1]  # Number of CSV files

        # Create rolling sequences
        for i in range(len(multivariate_data) - sequence_length):
            self.sequences.append(multivariate_data[i: i + sequence_length])

        # Convert to tensor
        self.sequences = torch.tensor(self.sequences, dtype=torch.float32)

        # Split the sequences into training and testing
        train_size = int(len(self.sequences) * (1 - test_size))
        self.train_data, self.test_data = torch.split(self.sequences, [train_size, len(self.sequences) - train_size])

        # Split timestamps into train and test
        self.timestamps_train = [ts[:train_size] for ts in self.timestamps]  # Split training timestamps
        self.timestamps_test = [ts[train_size:] for ts in self.timestamps]   # Split testing timestamps

        # Create expanded feature names with engineering features
        expanded_feature_names = []
        for feature in self.file_names:
            for eng_feature in features:
                expanded_feature_names.append(f"{feature} - {eng_feature}")

        self.feature_names = expanded_feature_names  # Store expanded names

        # Define DataLoaders
        self.train_dataloader = DataLoader(self.train_data, batch_size=32, shuffle=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=32, shuffle=False)

    def get_timestamps(self):
        return self.timestamps

    def __len__(self):
        return len(self.train_data)  # Return length of training data

    def __getitem__(self, idx):
        return self.train_data[idx]  # Return a training sequence

    def get_test_data(self):
        return self.test_data  # Return test data
    