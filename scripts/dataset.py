import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Series_Dataset(Dataset):
    def __init__(self, data, past_len, pred_len, stride=1):
        super(Series_Dataset, self).__init__()
        self.past_len = past_len
        self.pred_len = pred_len
        self.stride = stride
        self.df = data

        self.x_values, self.y_shifted, self.y_label, self.x_times, self.y_shifted_times, self.y_label_times = \
            self.create_sequences_with_cores_time(self.df, self.past_len, self.pred_len)

    def create_sequences_with_cores_time(
            self, data: pd.DataFrame, 
            past_len: int, pred_len: int
        ):
        
        x_values, y_shifted, y_label = [], [], []
        x_hours, y_shifted_hours, y_label_hours = [], [], []
        x_days, y_shifted_days, y_label_days = [], [], []
        x_months, y_shifted_months, y_label_months = [], [], []

        for i in range(0, len(data) - past_len - pred_len + 1):
            x_seq = data.iloc[i:i + past_len]
            y_shifted_seq = data.iloc[i+past_len-1 : i+past_len+pred_len-1]
            y_label_seq = data.iloc[i + past_len:i + past_len + pred_len]

            # Values
            x_values.append(x_seq['Waterlevel'].values)
            y_shifted.append(y_shifted_seq['Waterlevel'].values)
            y_label.append(y_label_seq['Waterlevel'].values)

            # Time features
            x_hours.append(x_seq['Hour'].values)
            y_shifted_hours.append(y_shifted_seq['Hour'].values)
            y_label_hours.append(y_label_seq['Hour'].values)

            x_days.append(x_seq['Day'].values)
            y_shifted_days.append(y_shifted_seq['Day'].values)
            y_label_days.append(y_label_seq['Day'].values)

            x_months.append(x_seq['Month'].values)
            y_shifted_months.append(y_shifted_seq['Month'].values)
            y_label_months.append(y_label_seq['Month'].values)

        x_values = np.array(x_values)
        y_shifted = np.array(y_shifted)
        y_label = np.array(y_label)

        x_times = np.stack([
            np.array(x_hours, dtype=np.int32),
            np.array(x_days, dtype=np.int32),
            np.array(x_months, dtype=np.int32)
        ], axis=-1)

        y_shifted_times = np.stack([
            np.array(y_shifted_hours, dtype=np.int32),
            np.array(y_shifted_days, dtype=np.int32),
            np.array(y_shifted_months, dtype=np.int32)
        ], axis=-1)

        y_label_times = np.stack([
            np.array(y_label_hours, dtype=np.int32),
            np.array(y_label_days, dtype=np.int32),
            np.array(y_label_months, dtype=np.int32)
        ], axis=-1)
        
        return (
            torch.tensor(x_values, dtype=torch.float32),
            torch.tensor(y_shifted, dtype=torch.float32),
            torch.tensor(y_label, dtype=torch.float32),
            torch.tensor(x_times, dtype=torch.int),
            torch.tensor(y_shifted_times, dtype=torch.int),
            torch.tensor(y_label_times, dtype=torch.int)
        )

    def __len__(self):
        return len(self.x_values)

    def __getitem__(self, idx):
        x_values = self.x_values[idx].unsqueeze(-1)
        y_shifted = self.y_shifted[idx].unsqueeze(-1)
        y_label = self.y_label[idx].unsqueeze(-1)
        
        x_times = self.x_times[idx]
        y_shifted_times = self.y_shifted_times[idx]
        y_label_times = self.y_label_times[idx]

        return x_values, y_shifted, y_label, x_times, y_shifted_times, y_label_times