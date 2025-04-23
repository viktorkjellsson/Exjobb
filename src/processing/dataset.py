from pathlib import Path
import pandas as pd
import numpy as np
import sys
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import MinMaxScaler

# Add the root project directory to the Python path
ROOT = Path.cwd().parent.parent
sys.path.append(str(ROOT))
from src.processing import preprocessing


class StrainDataset(Dataset):
    def __init__(self, folder_path, features, sequence_length, start_idx, test_size):
        self.train_sequences = []
        self.test_sequences = []
        self.timestamps_train = []
        self.timestamps_test = []
        self.file_names = []
        self.scaler = MinMaxScaler()
        self.sequence_length = sequence_length
        self.features = features

        self.per_file_train_scaled = []
        self.per_file_test_scaled = []
        self.per_file_train_timestamps = []
        self.per_file_test_timestamps = []

        all_train_features = []
        all_test_features = []
        all_train_timestamps = []
        all_test_timestamps = []

        for file in folder_path.glob("*.csv"):
            self.file_names.append(file.stem)
            df = pd.read_csv(file)
            df = df.iloc[start_idx:-1]
            df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

            timestamps = df["Time"].values
            strain_data = df.copy()

            split_idx = len(strain_data) - int(len(strain_data) * test_size)
            train_df = strain_data.iloc[:split_idx].reset_index(drop=True)
            test_df = strain_data.iloc[split_idx:].reset_index(drop=True)

            train_timestamps = timestamps[:split_idx]
            test_timestamps = timestamps[split_idx:]

            train_df = preprocessing.preprocessing_pipeline(train_df, interpolate_threshold=60)
            test_df = preprocessing.preprocessing_pipeline(test_df, interpolate_threshold=60)

            train_df = preprocessing.add_features(train_df, column="Strain")
            test_df = preprocessing.add_features(test_df, column="Strain")

            train_features = train_df[features].fillna(0).to_numpy()
            test_features = test_df[features].fillna(0).to_numpy()

            # Store raw train/test for global scaling
            all_train_features.append(train_features)
            all_test_features.append(test_features)
            all_train_timestamps.append(train_timestamps)
            all_test_timestamps.append(test_timestamps)

            # Store per-file scaled versions using individual scalers
            scaler_tmp = MinMaxScaler()
            scaler_tmp.fit(train_features)
            train_scaled_i = scaler_tmp.transform(train_features)
            test_scaled_i = scaler_tmp.transform(test_features)

            self.per_file_train_scaled.append(train_scaled_i)
            self.per_file_test_scaled.append(test_scaled_i)
            self.per_file_train_timestamps.append(train_timestamps)
            self.per_file_test_timestamps.append(test_timestamps)

        # Concatenate all features for global normalization
        all_train_features_cat = np.concatenate(all_train_features, axis=1)
        all_test_features_cat = np.concatenate(all_test_features, axis=1)
        all_train_timestamps_cat = np.concatenate(all_train_timestamps)
        all_test_timestamps_cat = np.concatenate(all_test_timestamps)

        self.scaler.fit(all_train_features_cat)
        train_scaled = self.scaler.transform(all_train_features_cat)
        test_scaled = self.scaler.transform(all_test_features_cat)

        # Rolling window creation (train)
        for i in range(len(train_scaled) - sequence_length):
            self.train_sequences.append(train_scaled[i: i + sequence_length])
            self.timestamps_train.append(all_train_timestamps_cat[i + sequence_length - 1])

        # Rolling window creation (test)
        for i in range(len(test_scaled) - sequence_length):
            self.test_sequences.append(test_scaled[i: i + sequence_length])
            self.timestamps_test.append(all_test_timestamps_cat[i + sequence_length - 1])

        # Expanded feature names for multivariate case
        self.feature_names = []
        for file in self.file_names:
            for feat in features:
                self.feature_names.append(f"{file} - {feat}")

        self.feature_count = len(self.feature_names)

        self.train_data = torch.tensor(np.array(self.train_sequences, dtype=np.float32))
        self.test_data = torch.tensor(np.array(self.test_sequences, dtype=np.float32))

        self.train_dataloader = DataLoader(self.train_data, batch_size=32, shuffle=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=32, shuffle=False)

    def get_timestamps(self):
        return self.timestamps_train, self.timestamps_test

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx]

    def get_test_data(self):
        return self.test_data
