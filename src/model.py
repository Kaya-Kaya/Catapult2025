import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PoseScoringModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(PoseScoringModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, X):
        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(device)
        out, _ = self.gru(X, h0)
        # out: batch x time x hidden
        out = out[:, -1, :]
        # out: batch x hidden
        out = self.fc(out)
        return out