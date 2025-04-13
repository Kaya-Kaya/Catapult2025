import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from typing import Tuple

class PoseData(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        self.input = []
        self.labels = []
        
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.mat'):
                    mat_data = loadmat(os.path.join(root, file))
                    self.input.append(mat_data)

                    index_str = file[file.rindex("_"):-4]
                    label_data = loadmat(os.path.join(root, f"metric_{index_str}.mat"))
                    self.labels.append(label_data)

        self.input = torch.tensor(self.input, device=self.device)
        self.labels = torch.tensor(self.labels, device=self.device)

    def __len__(self) -> int:
        return self.input.shape[0]
    

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input[idx], self.labels[idx]