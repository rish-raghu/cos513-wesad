from torch.utils.data import Dataset
import pandas as pd
import torch

ATTRIBUTES = [
    'chest__ACC_0', 
    # 'chest__ACC_1', 
    # 'chest__ACC_2', 
    # 'chest__ECG', 
    # 'chest__EMG', 
    # 'chest__EDA', 
    # 'chest__Temp', 
    # 'chest__Resp'
]

class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path, window):
        self.window = window
        csv = pd.read_csv(csv_path, compression="gzip")
        self.data = torch.tensor(csv[ATTRIBUTES].values).float()
        self.labels = torch.Tensor(csv['label']).int()
        print(f"Loaded {len(self.labels)} timesteps")

    def __len__(self):
        return len(self.labels) - self.window

    def __getitem__(self, index):
        return self.data[index:index+self.window]    
