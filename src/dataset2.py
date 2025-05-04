from torch.utils.data import Dataset
import pandas as pd
import torch

ATTRIBUTES = [
    'chest__ACC_0', 
    'chest__ACC_1', 
    'chest__ACC_2', 
    'chest__ECG', 
    'chest__EMG', 
    'chest__EDA', 
    'chest__Temp', 
    'chest__Resp'
]

class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path,subject_ids, window, stride=1):
        self.window = window
        data = pd.read_csv(csv_path, compression="gzip")
        data = data[data['subject'].isin(subject_ids)]
        data[ATTRIBUTES] = data.groupby('subject')[ATTRIBUTES].transform(lambda x: (x - x.mean()) / x.std())
        segment_ids = data.groupby('subject', sort=False, group_keys=True)['label'].apply(lambda x: (x != x.shift()).cumsum())
        data['segment_id'] = segment_ids.reset_index(level=0, drop=True)
        group_lens = data.groupby(['subject', 'segment_id'], sort=False).size().values
        start_idxs = []
        for i, length in enumerate(group_lens):
            group_start = group_lens[:i].sum()
            start_idxs.append(torch.arange(group_start, group_start + length - window, stride))
        self.start_idxs = torch.cat(start_idxs)
        self.data = torch.tensor(data[ATTRIBUTES].values).float()
        self.labels = torch.tensor(data['label'].values).int()
        print(f"Loaded {len(self.labels)} timesteps", flush=True)
        print(f"{len(self.start_idxs)} windows", flush=True)

    def __len__(self):
        return len(self.start_idxs)

    def __getitem__(self, index):
        start_idx = self.start_idxs[index]
        return self.data[start_idx:start_idx+self.window], self.labels[start_idx:start_idx+self.window]