import os
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class UCF50EncodedDataset(Dataset):
    def __init__(self, encoded_dir):
        self.encoded_dir = encoded_dir
        self.data_files = os.listdir(encoded_dir)
        self.frames = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        for data_file in self.data_files:
            file_path = os.path.join(self.encoded_dir, data_file)
            try:
                with h5py.File(file_path, 'r') as f:
                    self.frames.extend(f['frames'][:])
                    self.labels.extend(f['labels'][:])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]
        return torch.tensor(frame, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
