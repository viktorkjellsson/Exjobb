from pathlib import Path
from dataclasses import dataclass
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
    def __init__(self, folder_path, INPUT_FEATURES, OUTPUT_FEATURES, sequence_length, start_idx, test_size):
        print("Initializing StrainDataset...")

        self.file_names = []  # Store file names
        train_data_list = []
        test_data_list = []
        train_timestamps = []
        test_timestamps = []

        # Initialize a dictionary to store a MinMaxScaler for each feature
        scalers = {feature: MinMaxScaler() for feature in INPUT_FEATURES}

        for file in folder_path.glob("*.csv"):
            print(f"Processing file: {file}")
            self.file_names.append(file.stem)
            df = pd.read_csv(file)

            # Initial trimming
            df = df.iloc[start_idx:-1].copy()
            df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

            # Apply preprocessing
            df = preprocessing.preprocessing_pipeline(df, interpolate_threshold=0)

            # Split BEFORE feature engineering
            split_idx = int(len(df) * (1 - test_size))
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()

            # Save timestamps
            train_timestamps.extend(train_df["Time"].values)
            test_timestamps.extend(test_df["Time"].values)

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
        print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")

        # Apply MinMaxScaler to each feature independently
        train_scaled = np.zeros_like(train_data)
        test_scaled = np.zeros_like(test_data)
        print(f"Train scaled shape: {train_scaled.shape}, Test scaled shape: {test_scaled.shape}")


        for i, feature in enumerate(INPUT_FEATURES):
            # Fit the scaler on the training data for each feature
            train_scaled[:, i] = scalers[feature].fit_transform(train_data[:, i].reshape(-1, 1)).flatten()
            test_scaled[:, i] = scalers[feature].transform(test_data[:, i].reshape(-1, 1)).flatten()

        print(f"Train scaled shape after scaling: {train_scaled.shape}, Test scaled shape after scaling: {test_scaled.shape}")

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
        self.train_dataloader = DataLoader(self.train_data, batch_size=32, shuffle=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=32, shuffle=False)

        # Overlap check via hashing
        train_hashes = [hash(tuple(seq.numpy().flatten())) for seq in self.train_data]
        test_hashes = [hash(tuple(seq.numpy().flatten())) for seq in self.test_data]
        overlap_hashes = set(train_hashes).intersection(set(test_hashes))
        print(f"Number of overlapping hashes: {len(overlap_hashes)}")
        if overlap_hashes:
            print("Overlapping sequences detected! Potential data leakage.")
        else:
            print("No overlapping sequences detected. Train-test split is clean.")

        # Feature name expansion
        expanded_feature_names = []
        for sensor_name in self.file_names:
            for f in INPUT_FEATURES:
                expanded_feature_names.append(f"{sensor_name} - {f}")
        self.feature_names = expanded_feature_names
        self.feature_count = train_data.shape[1]

        # Store input and output feature names with sensor prefixes
        self.input_feature_names = []
        self.output_feature_names = []

        for sensor_name in self.file_names:
            self.input_feature_names.extend([f"{sensor_name} - {f}" for f in INPUT_FEATURES])
            self.output_feature_names.extend([f"{sensor_name} - {f}" for f in OUTPUT_FEATURES])

        # Optional: keep a generic feature_names alias for backwards compatibility
        self.feature_names = self.input_feature_names
        self.input_feature_count = len(self.input_feature_names)
        self.output_feature_count = len(self.output_feature_names)

    def get_timestamps(self):
        return self.timestamps_train, self.timestamps_test

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx]



# 2025-04-22 safe split, feature engineering still on the entire dataset
# from pathlib import Path
# import pandas as pd
# import numpy as np
# import sys
# from torch.utils.data import Dataset, DataLoader
# import torch

# # Add the root project directory to the Python path
# ROOT = Path.cwd().parent.parent
# sys.path.append(str(ROOT))
# from src.processing import preprocessing


# class StrainDataset(Dataset):
#     def __init__(self, folder_path, features, sequence_length, start_idx, test_size):
#         self.sequence_length = sequence_length
#         self.file_names = []
#         self.timestamps_train = []
#         self.timestamps_test = []

#         multivariate_data = []
#         all_timestamps = []

#         # Load and process each .csv file
#         for file in folder_path.glob("*.csv"):
#             self.file_names.append(file.stem)
#             df = pd.read_csv(file)
#             df = df.iloc[start_idx:-1]

#             df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
#             all_timestamps.append(df["Time"].values)

#             # Preprocessing and feature engineering
#             processed_df = preprocessing.preprocessing_pipeline(df, interpolate_threshold=60)
#             processed_df = preprocessing.add_features(processed_df, column="Strain")

#             strain_series = processed_df[features].fillna(0).to_numpy()
#             multivariate_data.append(strain_series)

#         # Combine data and timestamps
#         multivariate_data = np.concatenate(multivariate_data, axis=1)  # shape: [time, total_features]
#         self.feature_count = multivariate_data.shape[1]

#         merged_timestamps = np.concatenate(all_timestamps, axis=0)  # same length as multivariate_data

#         # Train/test split before sequence generation
#         num_timesteps = multivariate_data.shape[0]
#         split_index = int(num_timesteps * (1 - test_size))

#         train_data_raw = multivariate_data[:split_index]
#         test_data_raw = multivariate_data[split_index - sequence_length:]

#         train_timestamps_raw = merged_timestamps[:split_index]
#         test_timestamps_raw = merged_timestamps[split_index - sequence_length:]

#         # Create sequences for training
#         self.train_data = torch.tensor([
#             train_data_raw[i:i+sequence_length]
#             for i in range(len(train_data_raw) - sequence_length)
#         ], dtype=torch.float32)

#         self.timestamps_train = [
#             train_timestamps_raw[i:i+sequence_length]
#             for i in range(len(train_timestamps_raw) - sequence_length)
#         ]

#         # Create sequences for testing
#         self.test_data = torch.tensor([
#             test_data_raw[i:i+sequence_length]
#             for i in range(len(test_data_raw) - sequence_length)
#         ], dtype=torch.float32)

#         self.timestamps_test = [
#             test_timestamps_raw[i:i+sequence_length]
#             for i in range(len(test_timestamps_raw) - sequence_length)
#         ]

#         # Expanded feature names
#         expanded_feature_names = []
#         for file in self.file_names:
#             for feat in features:
#                 expanded_feature_names.append(f"{file} - {feat}")
#         self.feature_names = expanded_feature_names

#         # Define dataloaders
#         self.train_dataloader = DataLoader(self.train_data, batch_size=32, shuffle=True)
#         self.test_dataloader = DataLoader(self.test_data, batch_size=32, shuffle=False)

#     def get_timestamps(self):
#         return self.timestamps_train, self.timestamps_test

#     def __len__(self):
#         return len(self.train_data)

#     def __getitem__(self, idx):
#         return self.train_data[idx]

#     def get_test_data(self):
#         return self.test_data

# from pathlib import Path
# from dataclasses import dataclass
# import pandas as pd
# import numpy as np
# import sys
# import torch
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import MinMaxScaler

# # Add the root project directory to the Python path
# ROOT = Path.cwd().parent.parent  # Adjust if needed
# sys.path.append(str(ROOT))
# from src.processing import preprocessing


# class StrainDataset(Dataset):
#     def __init__(self, folder_path, features, sequence_length, start_idx, test_size):
#         print("Initializing StrainDataset...")

#         self.file_names = []  # Store file names
#         all_timestamps = []
#         multivariate_data = []

#         # Load and process each .csv file
#         for file in folder_path.glob("*.csv"):
#             print(f"Processing file: {file}")
#             self.file_names.append(file.stem)
#             df = pd.read_csv(file)

#             # Trim data
#             df = df.iloc[start_idx:-1]

#             # Convert timestamps
#             df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
#             all_timestamps.extend(df["Time"].values)  # Collect all timestamps as a single list

#             # Apply preprocessing pipeline
#             processed_df = preprocessing.preprocessing_pipeline(df, interpolate_threshold=60)

#             # Apply feature engineering
#             processed_df = preprocessing.add_features(processed_df, column="Strain")

#             # Extract features
#             strain_series = processed_df[features].fillna(0).to_numpy()
#             multivariate_data.append(strain_series)

#         # Concatenate data along the feature axis
#         multivariate_data = np.concatenate(multivariate_data, axis=1)
#         all_timestamps = np.array(all_timestamps)  # Convert timestamps to NumPy
#         print(f"Shape of concatenated multivariate data: {multivariate_data.shape}")

#         # Verify timestamps alignment
#         if len(all_timestamps) < multivariate_data.shape[0]:
#             raise ValueError("The number of timestamps does not match the expected number based on data rows.")

#         # Train-test split
#         train_size = int(len(multivariate_data) * (1 - test_size))
#         train_data = multivariate_data[:train_size]
#         test_data = multivariate_data[train_size:]
#         train_timestamps = all_timestamps[:train_size]
#         test_timestamps = all_timestamps[train_size:]

#         # Create rolling sequences for train and test
#         train_sequences = [train_data[i: i + sequence_length] for i in range(len(train_data) - sequence_length)]
#         test_sequences = [test_data[i: i + sequence_length] for i in range(len(test_data) - sequence_length)]

#         # Convert rolling sequences into NumPy arrays
#         train_sequences_np = np.array(train_sequences)
#         test_sequences_np = np.array(test_sequences)

#         # Apply scaling after splitting to prevent data leakage
#         scaler = MinMaxScaler()
#         train_sequences_np = scaler.fit_transform(train_sequences_np.reshape(train_sequences_np.shape[0], -1))
#         test_sequences_np = scaler.transform(test_sequences_np.reshape(test_sequences_np.shape[0], -1))

#         # Convert back to tensors correctly
#         self.train_data = torch.tensor(train_sequences_np.reshape(train_sequences_np.shape[0], sequence_length, -1), dtype=torch.float32)
#         self.test_data = torch.tensor(test_sequences_np.reshape(test_sequences_np.shape[0], sequence_length, -1), dtype=torch.float32)

#         # Assign rolling timestamps
#         self.timestamps_train = train_timestamps
#         self.timestamps_test = test_timestamps

#         # Check for overlaps
#         train_hashes = [hash(tuple(seq.numpy().flatten())) for seq in self.train_data]
#         test_hashes = [hash(tuple(seq.numpy().flatten())) for seq in self.test_data]
#         overlap_hashes = set(train_hashes).intersection(set(test_hashes))
#         print(f"Number of overlapping hashes: {len(overlap_hashes)}")
#         if len(overlap_hashes) > 0:
#             print("Overlapping sequences detected! Potential data leakage.")
#         else:
#             print("No overlapping sequences detected. Train-test split is clean.")

#         # # Check for index overlaps
#         # train_indices_set = set(self.train_indices)
#         # test_indices_set = set(self.test_indices)
#         # overlap_indices = train_indices_set.intersection(test_indices_set)
#         # print(f"Number of overlapping indices: {len(overlap_indices)}")
#         # if len(overlap_indices) > 0:
#         #     print("Overlapping indices detected! Potential data leakage.")
#         # else:
#         #     print("No overlapping indices detected. Train-test split is clean.")

#         # Define DataLoaders
#         self.train_dataloader = DataLoader(self.train_data, batch_size=32, shuffle=True)
#         self.test_dataloader = DataLoader(self.test_data, batch_size=32, shuffle=False)

#         # Feature count
#         self.feature_count = multivariate_data.shape[1]

#         # Expanded feature names
#         expanded_feature_names = []
#         for feature in self.file_names:
#             for eng_feature in features:
#                 expanded_feature_names.append(f"{feature} - {eng_feature}")
#         self.feature_names = expanded_feature_names

#     def get_timestamps(self):
#         return self.timestamps_train, self.timestamps_test

#     def __len__(self):
#         return len(self.train_data)

#     def __getitem__(self, idx):
#         return self.train_data[idx]




#Original code:
# from pathlib import Path
# from dataclasses import dataclass
# import pandas as pd
# import numpy as np
# import sys
# from torch.utils.data import Dataset, DataLoader
# import torch

# # Add the root project directory to the Python path
# ROOT = Path.cwd().parent.parent  # This will get the project root since the notebook is in 'notebooks/'
# sys.path.append(str(ROOT))
# from src.processing import preprocessing

# class StrainDataset(Dataset):
#     def __init__(self, folder_path, features, sequence_length, start_idx, test_size):
#         self.sequences = []
#         self.timestamps = []
#         self.timestamps_train = []
#         self.timestamps_test = []  
#         self.file_names = []  # Store file names without .csv

#         multivariate_data = []

#         # Load and process each .csv file in the folder
#         for file in folder_path.glob("*.csv"):
#             self.file_names.append(file.stem)
#             df = pd.read_csv(file)
#             df = df.iloc[start_idx:-1]  # Trim data based on start index
            
#             df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
#             self.timestamps.append(df["Time"].values)  # Store timestamps
            
#             # Apply preprocessing pipeline
#             processed_df = preprocessing.preprocessing_pipeline(df, interpolate_threshold=60)  # Process the data
            
#             # Apply feature engineering
#             processed_df = preprocessing.add_features(processed_df, column="Strain")

#             strain_series = processed_df[features].fillna(0).to_numpy()
#             multivariate_data.append(strain_series)  # Append the data from each file
        
#         # After the loop, concatenate all the data
#         multivariate_data = np.concatenate(multivariate_data, axis=1)  # Concatenate along the correct axis
#         self.feature_count = multivariate_data.shape[1]  # Number of CSV files

#         # Create rolling sequences
#         for i in range(len(multivariate_data) - sequence_length):
#             self.sequences.append(multivariate_data[i: i + sequence_length])

#         # Convert to tensor
#         self.sequences = torch.tensor(self.sequences, dtype=torch.float32)

#         # Split the sequences into training and testing
#         train_size = int(len(self.sequences) * (1 - test_size))
#         self.train_data, self.test_data = torch.split(self.sequences, [train_size, len(self.sequences) - train_size])

#         # Split timestamps into train and test
#         self.timestamps_train = [ts[:train_size] for ts in self.timestamps]  # Split training timestamps
#         self.timestamps_test = [ts[train_size:] for ts in self.timestamps]   # Split testing timestamps

#         # Create expanded feature names with engineering features
#         expanded_feature_names = []
#         for feature in self.file_names:
#             for eng_feature in features:
#                 expanded_feature_names.append(f"{feature} - {eng_feature}")

#         self.feature_names = expanded_feature_names  # Store expanded names

#         # Define DataLoaders
#         self.train_dataloader = DataLoader(self.train_data, batch_size=32, shuffle=True)
#         self.test_dataloader = DataLoader(self.test_data, batch_size=32, shuffle=False)

#     def get_timestamps(self):
#         return self.timestamps

#     def __len__(self):
#         return len(self.train_data)  # Return length of training data

#     def __getitem__(self, idx):
#         return self.train_data[idx]  # Return a training sequence

#     def get_test_data(self):
#         return self.test_data  # Return test data
    

    ######################################################
    

# from pathlib import Path
# from dataclasses import dataclass
# import pandas as pd
# import numpy as np
# import sys
# from torch.utils.data import Dataset, DataLoader
# import torch

# # Add the root project directory to the Python path
# ROOT = Path.cwd().parent.parent  # This will get the project root since the notebook is in 'notebooks/'
# sys.path.append(str(ROOT))
# from src.processing import preprocessing

# class StrainDataset(Dataset):
#     def __init__(self, folder_path, features, sequence_length, start_idx, test_size):
#         print("Initializing StrainDataset...")
#         self.sequences = []
#         self.timestamps = []
#         self.timestamps_train = []
#         self.timestamps_test = []  
#         self.file_names = []  # Store file names without .csv

#         multivariate_data = []

#         # Load and process each .csv file in the folder
#         # for file in folder_path.glob("*.csv"):
#         #     print(f"Processing file: {file}")  # Debugging file loading
#         #     self.file_names.append(file.stem)
#         #     df = pd.read_csv(file)
#         #     df = df.iloc[start_idx:-1]  # Trim data based on start index
            
#         #     self.timestamps = df["Time"].values  # One-dimensional array
#         for i, file in enumerate(folder_path.glob("*.csv")):
#             df = pd.read_csv(file)
#             df = df.iloc[start_idx:-1]
    
#             if i == 0:
#                 df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
#                 self.timestamps = df["Time"]  # Store timestamps from the first file
#                 print(f"Loaded timestamps from {file}: {self.timestamps[:5]}")  # Print first 5 timestamps for debugging
#                 print(f'Type of timestamps: {type(self.timestamps)}')

#             if len(self.timestamps) == 0:
#                 print("Timestamps are empty!")
#             else:
#                 print(f"Timestamps contain {len(self.timestamps)} entries.")
#                 print(self.timestamps[:5])  # Print the first 5 timestamps
                    
#             # Apply preprocessing pipeline
#             processed_df = preprocessing.preprocessing_pipeline(df, interpolate_threshold=60)  # Process the data
            
#             # Apply feature engineering
#             processed_df = preprocessing.add_features(processed_df, column="Strain")

#             strain_series = processed_df[features].fillna(0).to_numpy()
#             multivariate_data.append(strain_series)  # Append the data from each file
        
#         # After the loop, concatenate all the data
#         multivariate_data = np.concatenate(multivariate_data, axis=1)  # Concatenate along the correct axis
#         print(f"Shape of concatenated multivariate data: {multivariate_data.shape}")

#         # Train-test split
#         train_size = int(len(multivariate_data) * (1 - test_size))
#         train_data = multivariate_data[:train_size]
#         test_data = multivariate_data[train_size:]

#         # Create rolling sequences separately
#         train_sequences = [train_data[i: i + sequence_length] for i in range(len(train_data) - sequence_length)]
#         test_sequences = [test_data[i: i + sequence_length] for i in range(len(test_data) - sequence_length)]

#         # Align timestamps with sequence starts
#         self.timestamps_train = self.timestamps[:len(train_sequences)]
#         self.timestamps_test = self.timestamps[train_size:train_size + len(test_sequences)]
#         # self.timestamps_train = [ts[:train_size] for ts in self.timestamps]  # Split training timestamps
#         # self.timestamps_test = [ts[train_size:] for ts in self.timestamps]   # Split testing timestamps
#         # self.timestamps_train = self.timestamps[start_idx:start_idx + train_size - sequence_length]
#         # self.timestamps_test = self.timestamps[start_idx + train_size:start_idx + train_size + len(test_sequences)]
#         print(f"Shape of train sequences: {len(train_sequences)}, Shape of test sequences: {len(test_sequences)}")
#         print(f"Shape of train timestamps: {len(self.timestamps_train)}, Shape of test timestamps: {len(self.timestamps_test)}")
#         print(f"First 5 train timestamps: {self.timestamps_train[:5]}")
#         print(f"First 5 test timestamps: {self.timestamps_test[:5]}")
#         print(f'Type of train timestamps: {type(self.timestamps_train)}')
#         print(f'Type of test timestamps: {type(self.timestamps_test)}')

#         # Convert to tensors
#         self.train_data = torch.tensor(train_sequences, dtype=torch.float32)
#         self.test_data = torch.tensor(test_sequences, dtype=torch.float32)

#         # # Assign train and test timestamps
#         # self.timestamps_train = train_timestamps
#         # self.timestamps_test = test_timestamps

#         # Check for overlaps
#         train_hashes = [hash(tuple(seq.numpy().flatten())) for seq in self.train_data]
#         test_hashes = [hash(tuple(seq.numpy().flatten())) for seq in self.test_data]
#         overlap_hashes = set(train_hashes).intersection(set(test_hashes))
#         print(f"Number of overlapping hashes: {len(overlap_hashes)}")
#         if len(overlap_hashes) > 0:
#             print("Overlapping sequences detected! Potential data leakage.")
#         else:
#             print("No overlapping sequences detected. Train-test split is clean.")

#         # Define DataLoaders
#         self.train_dataloader = DataLoader(self.train_data, batch_size=32, shuffle=True)
#         self.test_dataloader = DataLoader(self.test_data, batch_size=32, shuffle=False)


#         self.feature_count = multivariate_data.shape[1]  # Number of CSV files

#         # # Create rolling sequences
#         # for i in range(len(multivariate_data) - sequence_length):
#         #     self.sequences.append(multivariate_data[i: i + sequence_length])

#         # # Convert to tensor
#         # self.sequences = torch.tensor(self.sequences, dtype=torch.float32)

#         # # Split the sequences into training and testing
#         # train_size = int(len(self.sequences) * (1 - test_size))
#         # self.train_data, self.test_data = torch.split(self.sequences, [train_size, len(self.sequences) - train_size])

#         # # Debugging: Check for overlaps between training and testing
#         # train_indices = [list(range(i, i + sequence_length)) for i in range(len(self.train_data))]
#         # test_indices = [list(range(i, i + sequence_length)) for i in range(len(self.test_data))]

#         # overlap = any(set(train_range).intersection(set(test_range)) for train_range in train_indices for test_range in test_indices)
#         # print(f"Is there an overlap between training and testing sequences? {overlap}")

#         # # Debugging: Check timestamps for overlaps (if available)
#         # train_timestamps = [ts[:train_size] for ts in self.timestamps]
#         # test_timestamps = [ts[train_size:] for ts in self.timestamps]

#         # train_set = set([timestamp for ts in train_timestamps for timestamp in ts])
#         # test_set = set([timestamp for ts in test_timestamps for timestamp in ts])
#         # overlap_timestamps = train_set.intersection(test_set)

#         # print(f"Number of overlapping timestamps: {len(overlap_timestamps)}")
#         # print(f"Overlapping timestamps: {overlap_timestamps}")

#         # # Split timestamps into train and test
#         # self.timestamps_train = [ts[:train_size] for ts in self.timestamps]  # Split training timestamps
#         # self.timestamps_test = [ts[train_size:] for ts in self.timestamps]   # Split testing timestamps

#         # Create expanded feature names with engineering features
#         expanded_feature_names = []
#         for feature in self.file_names:
#             for eng_feature in features:
#                 expanded_feature_names.append(f"{feature} - {eng_feature}")

#         self.feature_names = expanded_feature_names  # Store expanded names

#         # # Define DataLoaders
#         # self.train_dataloader = DataLoader(self.train_data, batch_size=32, shuffle=True)
#         # self.test_dataloader = DataLoader(self.test_data, batch_size=32, shuffle=False)

#     # def get_timestamps(self):
#     #     return self.timestamps
    
#     # def get_train_timestamps(self):
#     #     return self.timestamps_train

#     # def get_test_timestamps(self):
#     #     return self.timestamps_test

#     def __len__(self):
#         return len(self.train_data)  # Return length of training data

#     def __getitem__(self, idx):
#         return self.train_data[idx]  # Return a training sequence

#     def get_test_data(self):
#         return self.test_data  # Return test data
    