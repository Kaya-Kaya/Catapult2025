import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from typing import Tuple
import random
import numpy as np

class PoseData(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input = sorted([os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.startswith("data") and f.endswith(".mat")])
        self.labels = sorted([os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.startswith("metric") and f.endswith(".mat")])
        
    def shuffle(self) -> None:
        combined = list(zip(self.input, self.labels))
        random.shuffle(combined)
        self.input, self.labels = zip(*combined)

    def __len__(self) -> int:
        return len(self.input)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(idx, slice):
            mat_data = [loadmat(self.input[i]) for i in range(*idx.indices(len(self.input)))]
            label_data = [loadmat(self.labels[i]) for i in range(*idx.indices(len(self.labels)))]
            mat_data = [data['pose'] for data in mat_data]
            label_data = [data['metric'] for data in label_data]
            return (
                torch.tensor(np.array(mat_data), device=self.device),
                torch.tensor(np.array(label_data), device=self.device),
            )
        else:
            mat_data = loadmat(self.input[idx])
            label_data = loadmat(self.labels[idx])
            return torch.tensor(mat_data, device=self.device), torch.tensor(label_data, device=self.device)