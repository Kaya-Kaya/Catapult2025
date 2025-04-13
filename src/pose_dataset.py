import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from typing import Tuple
import numpy as np

class PoseData(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input = sorted([os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.startswith("data") and f.endswith(".mat")])
        self.labels = sorted([os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.startswith("metric") and f.endswith(".mat")])

    def __len__(self) -> int:
        return len(self.input)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        mat_data = loadmat(self.input[idx])
        label_data = loadmat(self.labels[idx])
        
        # Replace 'data' and 'label' with the actual variable names in your .mat files
        # For example, if your main variable in the input file is stored under the key "data"
        # and in the label file under "label"
        data_array = mat_data['pose']      # Adjust key name as necessary
        label_array = label_data['metric']   # Adjust key name as necessary

        # Convert the arrays to PyTorch tensors, specifying dtype if needed
        data_tensor = torch.tensor(data_array, dtype=torch.float32, device=self.device)
        label_tensor = torch.tensor(label_array, dtype=torch.float32, device=self.device)
 
        return data_tensor, label_tensor
