import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from typing import Tuple

class PoseData(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self) -> int:
        return self.input.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        mat_data = loadmat(os.path.join(self.root_dir, f"data_{str(idx).zfill(3)}.mat"))

        label_data = loadmat(os.path.join(self.root_dir, f"metric_{str(idx).zfill(3)}.mat"))

        return torch.tensor(mat_data, device=self.device), torch.tensor(label_data, device=self.device)