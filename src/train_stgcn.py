import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import scipy.io
import numpy as np
from tqdm import tqdm
import sys
import math
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# Filter UndefinedMetricWarning (happens if a class has no predictions/true samples in a batch/epoch)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# --- Graph Definition ---
num_nodes = 33 # 33 landmarks
# Connections based on MediaPipe Pose landmarks (indices 0-32)
skeleton_edges = [
    # Face / Head
    (0, 1), (1, 2), (2, 3), (3, 7), # Left eye area -> ear
    (0, 4), (4, 5), (5, 6), (6, 8), # Right eye area -> ear
    (9, 10), # Mouth line

    # Body / Torso
    (11, 12), # Shoulder line
    (11, 23), # Left shoulder to left hip
    (12, 24), # Right shoulder to right hip
    (23, 24), # Hip line

    # Left Arm
    (11, 13), # Shoulder to elbow
    (13, 15), # Elbow to wrist
    (15, 17), # Wrist to pinky
    (15, 19), # Wrist to index
    (15, 21), # Wrist to thumb
    (17, 19), # Pinky to index (optional, represents palm edge)

    # Right Arm
    (12, 14), # Shoulder to elbow
    (14, 16), # Elbow to wrist
    (16, 18), # Wrist to pinky
    (16, 20), # Wrist to index
    (16, 22), # Wrist to thumb
    (18, 20), # Pinky to index (optional, represents palm edge)

    # Left Leg
    (23, 25), # Hip to knee
    (25, 27), # Knee to ankle
    (27, 29), # Ankle to heel
    (27, 31), # Ankle to foot index (toe)
    (29, 31), # Heel to foot index (toe)

    # Right Leg
    (24, 26), # Hip to knee
    (26, 28), # Knee to ankle
    (28, 30), # Ankle to heel
    (28, 32), # Ankle to foot index (toe)
    (30, 32), # Heel to foot index (toe)
]

def build_adjacency_matrix(num_nodes, edges):
    """Builds a normalized adjacency matrix for GCN."""
    adj = torch.zeros((num_nodes, num_nodes))
    for i, j in edges:
        if 0 <= i < num_nodes and 0 <= j < num_nodes:
            adj[i, j] = 1
            adj[j, i] = 1 # Assuming undirected graph
        else:
            print(f"Warning: Edge index ({i}, {j}) out of bounds for {num_nodes} nodes.", file=sys.stderr)

    # Add self-loops
    identity = torch.eye(num_nodes)
    adj = adj + identity

    # Normalize adjacency matrix (Symmetric normalization)
    row_sum = adj.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    adj_normalized = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return adj_normalized

# --- Model Components ---

class GraphConvolution(nn.Module):
    """Simple GCN layer."""
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        # x shape: (Batch * Time, Nodes, Features_in)
        # adj shape: (Nodes, Nodes)
        support = torch.matmul(x, self.weight) # (B*T, N, F_out)
        # Use torch.bmm for batch matrix multiplication if adj varies per batch item (not the case here)
        # output = torch.matmul(adj, support) # Basic GCN propagation

        # More robust way for fixed adj: einsum or reshape + mm
        # Reshape for matmul: (N, N) x (N, B*T*F_out) -> needs transpose
        # output = torch.matmul(adj, support.permute(1,0,2).reshape(num_nodes,-1))
        # output = output.reshape(num_nodes, x.shape[0], -1).permute(1,0,2)

        # Einsum approach (often clearer for GCN)
        # 'bni, io -> bno' if input is (B*T, N, F_in)
        # 'uv, bvi -> bui' if input is (B*T, N, F_in) and adj is (N,N)
        output = torch.einsum('uv,bvi->bui', adj, support) # (B*T, N, F_out)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class STGCNBlock(nn.Module):
    """Spatio-temporal GCN block."""
    def __init__(self, in_features, out_features, temporal_kernel_size):
        super(STGCNBlock, self).__init__()
        self.in_features = in_features # Store in_features for residual check
        self.out_features = out_features
        self.gcn = GraphConvolution(in_features, out_features)
        # Temporal Conv: acts on features over time for each node independently initially
        # Input to Conv1d: (Batch, Channels=Nodes*Features, Time)
        # Or treat temporal conv as Conv2d: (Batch, Channels=Features, Time, Nodes)
        # Let's use Conv2d approach common in ST-GCN papers
        self.tcn = nn.Conv2d(out_features, out_features,
                             kernel_size=(temporal_kernel_size, 1), # Convolve over time (dim=2)
                             padding=((temporal_kernel_size - 1) // 2, 0)) # Pad time dim

        self.bn_gcn = nn.BatchNorm2d(num_nodes) # Apply BN after GCN -> (B*T, N, F) -> (B, N, F, T)? Needs reshape
        self.bn_tcn = nn.BatchNorm2d(out_features) # Apply BN after TCN (B, F, T, N)

        self.relu = nn.ReLU(inplace=True)

        # Residual connection handling
        if in_features != out_features:
            # Use Conv2d for residual to handle (B, C, T, N) format easily
            self.residual_conv = nn.Conv2d(in_features, out_features, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, adj):
        # x shape: (Batch, Features_in, Time, Nodes) - Adjusted expected input format
        B, C_in, T, N = x.shape # Note the change in expected input dimension order

        # --- Residual Calculation ---
        # Input to residual conv should be (B, C_in, T, N), which is x itself
        res = self.residual_conv(x)

        # --- Spatial (GCN) ---
        # Reshape for GCN: (B, C_in, T, N) -> (B*T, N, C_in) -> Needs permute
        x_permuted_for_gcn = x.permute(0, 2, 3, 1) # (B, T, N, C_in)
        x_gcn_in = x_permuted_for_gcn.reshape(B * T, N, C_in)
        x_gcn_out = self.gcn(x_gcn_in, adj) # (B*T, N, F_out)
        x_gcn_out_act = self.relu(x_gcn_out)

        # --- Temporal (TCN) ---
        # Reshape for Conv2d: (B*T, N, F_out) -> (B, T, N, F_out) -> (B, F_out, T, N)
        x_tcn_in = x_gcn_out_act.view(B, T, N, self.out_features).permute(0, 3, 1, 2)
        x_tcn_out = self.tcn(x_tcn_in) # (B, F_out, T, N)
        x_bn_out = self.bn_tcn(x_tcn_out)
        x_tcn_out_act = self.relu(x_bn_out)

        # Add residual (both are (B, F_out, T, N))
        x_final = x_tcn_out_act + res

        return x_final # Return as (B, F_out, T, N)

# --- Full Model ---
class STGCNModel(nn.Module):
    def __init__(self, num_nodes, in_features, num_classes, temporal_kernel_size=9):
        super(STGCNModel, self).__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features # Should be 3 now
        self.num_classes = num_classes

        self.adj = build_adjacency_matrix(num_nodes, skeleton_edges)
        # Apply BN on feature dimension C = in_features
        # Input format for BN1d is (N, C) or (N, C, L), maybe easier to apply after reshape?
        # Let's use BatchNorm2d expecting (B, C, H, W) -> (B, C, T, N)
        self.bn_input = nn.BatchNorm2d(in_features)

        self.stgcn_block1 = STGCNBlock(in_features, 64, temporal_kernel_size)
        self.stgcn_block2 = STGCNBlock(64, 128, temporal_kernel_size)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        last_features = 128
        # --- FIX: Correct FC layer input size ---
        self.fc = nn.Linear(last_features, num_classes)
        # ----------------------------------------

    def forward(self, x):
        # x shape from DataLoader: (Batch, Time, Nodes, Features=3) = (B, T, N, C)
        B, T, N, C = x.shape
        adj = self.adj.to(x.device)

        # Permute for Conv Layers: (B, T, N, C) -> (B, C, T, N)
        x = x.permute(0, 3, 1, 2)

        # Apply initial batch norm: Input (B, C, T, N)
        x = self.bn_input(x)

        # ST-GCN Blocks (expect and return (B, F, T, N))
        x = self.stgcn_block1(x, adj) # (B, 64, T, N)
        x = self.stgcn_block2(x, adj) # (B, 128, T, N)

        # Pooling: Input (B, F, T, N)
        x_pooled = self.pool(x) # (B, F, 1, 1)

        # Flatten and FC layer
        x_flat = x_pooled.view(B, -1) # Flatten features: (B, F)
        x_out = self.fc(x_flat) # (B, num_classes)

        return torch.sigmoid(x_out)

# --- Dataset ---
class PoseDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
             raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        self.data_files = sorted(list(self.data_dir.glob("data_*.mat")))
        self.metric_files = sorted(list(self.data_dir.glob("metric_*.mat")))

        if len(self.data_files) != len(self.metric_files):
            print(f"Warning: Mismatch in number of data ({len(self.data_files)}) and metric ({len(self.metric_files)}) files.")
            # Basic matching logic (adjust if needed)
            data_indices = {f.stem.split('_')[-1] for f in self.data_files}
            metric_indices = {f.stem.split('_')[-1] for f in self.metric_files}
            valid_indices = data_indices.intersection(metric_indices)
            self.data_files = sorted([f for f in self.data_files if f.stem.split('_')[-1] in valid_indices])
            self.metric_files = sorted([f for f in self.metric_files if f.stem.split('_')[-1] in valid_indices])
            print(f"Using {len(self.data_files)} matched files.")

        if not self.data_files:
            raise FileNotFoundError(f"No matched data/metric files found in {self.data_dir}")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_path = self.data_files[idx]
        metric_path = self.metric_files[idx]
        try:
            pose_data_mat = scipy.io.loadmat(data_path)
            # --- FIX: Slice to keep only first 3 coordinates (X, Y, Z) ---
            pose_data = pose_data_mat['pose'][:, :, :3].astype(np.float32)
            # Check if slicing resulted in empty array (shouldn't happen if C>=3)
            if pose_data.shape[2] != 3:
                raise ValueError(f"Expected 3 coordinates after slicing, but got {pose_data.shape[2]} for {data_path.name}")
            # --------------------------------------------------------------
            metric_data_mat = scipy.io.loadmat(metric_path)
            metric_data = metric_data_mat['metric'].flatten().astype(np.float32)
            return torch.from_numpy(pose_data), torch.from_numpy(metric_data)
        except Exception as e:
            print(f"\nError loading file index {idx}: {data_path.name} / {metric_path.name}. Error: {e}", file=sys.stderr)
            if idx > 0: return self.__getitem__(0)
            else: raise RuntimeError(f"Failed to load first data item: {e}")


# --- Training --- (Modified to include validation and metrics)
def train_model(data_dir, num_epochs=50, batch_size=16, learning_rate=0.001, val_split=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate Dataset
    try:
        full_dataset = PoseDataset(data_dir)
        if len(full_dataset) == 0:
             print("Error: Dataset is empty. Check data directory and file matching.")
             return
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    # Split dataset
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Dataset split: Train={len(train_dataset)}, Validation={len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Instantiate Model
    try:
        sample_data, _ = full_dataset[0] # Get sample from full dataset before splitting
        in_features = sample_data.shape[2] # C dimension
        print(f"Detected input features (coordinates): {in_features}")
    except Exception as e:
        print(f"Error getting sample data to determine input features: {e}. Assuming 3 features.")
        in_features = 3 # Default

    num_classes = 9 # Output metrics
    model = STGCNModel(num_nodes=num_nodes, in_features=in_features, num_classes=num_classes).to(device)

    # Loss Function and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)

        for i, (inputs, labels) in enumerate(train_progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {epoch_loss:.4f}")

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)

        with torch.no_grad():
            for inputs, labels in val_progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Store predictions and labels for metrics calculation
                # Apply threshold (0.5) for precision/recall/f1
                preds = (outputs > 0.5).cpu().numpy()
                labels_np = labels.cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels_np)

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Validation Loss: {avg_val_loss:.4f}")

        # Concatenate all predictions and labels
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Calculate and Print Metrics per class
        print("--- Validation Metrics (Per Class) ---")
        try:
            # Precision, Recall, F1 (binary for each class, threshold=0.5)
            precision, recall, f1, support = precision_recall_fscore_support(
                all_labels, all_preds, average=None, labels=list(range(num_classes)), zero_division=0
            )

            # AUC-ROC and Average Precision (using probabilities)
            # Note: Need raw model outputs (before thresholding) for AUC scores
            # Let's re-run inference slightly differently for probabilities (less efficient but simpler here)
            all_probs = []
            with torch.no_grad():
                 for inputs, _ in val_loader: # Only need inputs now
                     inputs = inputs.to(device)
                     outputs = model(inputs)
                     all_probs.append(outputs.cpu().numpy())
            all_probs = np.concatenate(all_probs, axis=0)

            roc_auc = roc_auc_score(all_labels, all_probs, average=None)
            avg_prec = average_precision_score(all_labels, all_probs, average=None)

            header = f"{'Metric':<10}" + "".join([f'{i:<8}' for i in range(num_classes)])
            print(header)
            print("-" * len(header))
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1-Score:", f1)
            print("AUC-ROC:", roc_auc)
            print("Avg Precision:", avg_prec)
            print("Support:", support)
            print("-" * len(header))

        except ValueError as e:
             print(f"Could not calculate some metrics (likely due to lack of positive/negative samples for a class in validation set): {e}")
        except Exception as e:
             print(f"An error occurred during metric calculation: {e}")

    print("Finished Training")

    # --- Save the trained model (optional) ---
    # model_save_path = Path("models") / "stgcn_model.pth"
    # model_save_path.parent.mkdir(parents=True, exist_ok=True)
    # torch.save(model.state_dict(), model_save_path)
    # print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent # Workspace root
    data_directory = base_dir / "data_og"

    # --- Hyperparameters ---
    EPOCHS = 50
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2 # Fraction of data for validation

    train_model(data_directory, num_epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, val_split=VALIDATION_SPLIT) 