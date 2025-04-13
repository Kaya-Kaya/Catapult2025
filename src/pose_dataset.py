import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from typing import Tuple
import random

class PoseData(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input = sorted([f for f in os.listdir(self.root_dir) if f.startswith("data") and f.endswith(".mat")])
        self.labels = sorted([f for f in os.listdir(self.root_dir) if f.startswith("metric") and f.endswith(".mat")])
        
    def shuffle(self) -> None:
        combined = list(zip(self.input, self.labels))
        random.shuffle(combined)
        self.input, self.labels = zip(*combined)

    def __len__(self) -> int:
        return len(self.input)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        mat_data = loadmat(self.input[idx])
        label_data = loadmat(self.labels[idx])

        return torch.tensor(mat_data, device=self.device), torch.tensor(label_data, device=self.device)