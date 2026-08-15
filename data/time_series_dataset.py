import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, input, window):

        if isinstance(input, pd.DataFrame):
            self.x = torch.tensor(input.iloc[:, 1:].values, dtype=torch.float32)
            self.y = torch.tensor(input.iloc[:, [0]].values, dtype=torch.float32)
        elif isinstance(input, np.ndarray):
            self.x = torch.tensor(input[:, 1:], dtype=torch.float32)
            self.y = torch.tensor(input[:, [0]], dtype=torch.float32)
        else:
            raise TypeError("Input must be Pandas DataFrame or NumPy array.")

        self.window = window
        self.num_windows = len(input) - window + 1

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.window

        return self.x[start_idx:end_idx, :], self.y[start_idx:end_idx, :]
