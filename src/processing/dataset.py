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
from configs.path_config import GROUP1A, EXTRACTED_DATA_DIR
from src.processing import preprocessing

class StrainDataset(Dataset):
    def __init__(self, root, group, sequence_length, test_size=0.2):
        self.sequences = []

        multivariate_data = []
        # Load and process each file
        for file in group:
            path = EXTRACTED_DATA_DIR / 'group1a' / file
            df = pd.read_csv(path)
            df = df.iloc[12216:-1]
            processed_df = preprocessing.preprocessing_pipeline(df)
            strain_series = processed_df["Strain"].fillna(0)
            multivariate_data.append(strain_series)

        multivariate_data = np.stack(multivariate_data, axis=1)

        # Create rolling sequences
        for i in range(len(multivariate_data) - sequence_length):
            self.sequences.append(multivariate_data[i: i + sequence_length])

        # Convert to tensor
        self.sequences = torch.tensor(self.sequences, dtype=torch.float32)

        # Split the sequences into training and testing
        train_size = int(len(self.sequences) * (1 - test_size))
        self.train_data, self.test_data = torch.split(self.sequences, [train_size, len(self.sequences) - train_size])

        # Define DataLoaders
        self.train_dataloader = DataLoader(self.train_data, batch_size=32, shuffle=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=32, shuffle=False)

    def __len__(self):
        return len(self.train_data)  # Return length of training data

    def __getitem__(self, idx):
        return self.train_data[idx]  # Return a training sequence

    def get_test_data(self):
        return self.test_data  # Return test data 