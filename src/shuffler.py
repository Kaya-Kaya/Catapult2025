import os
import scipy.io
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm

# Configuration
# Get the directory containing the script
script_dir = Path(__file__).parent

# Construct paths relative to the script directory and resolve to absolute paths
BASE_DIR = (script_dir / "../Poses").resolve()
BAD_DIR = BASE_DIR / "Bad"
GOOD_DIR = BASE_DIR / "Good"
OUTPUT_DIR = BASE_DIR / "Shuffled"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Landmark groups
landmark_group = {
    0: list(range(11)),  # Head/face
    1: [11, 13, 15, 17, 19, 21],  # Left arm
    2: [12, 14, 16, 18, 20, 22],  # Right arm
    3: [11, 12, 23, 24],  # Torso
    4: [23, 24],  # Pelvis
    5: [23, 25, 27],  # Left leg
    6: [24, 26, 28],  # Right leg
    7: [27, 29, 31],  # Left foot
    8: [28, 30, 32]  # Right foot
}


def load_swings(directory):
    """Load .mat files with proper error handling"""
    swings = []
    if not directory.exists():
        # Use the resolved absolute path in the error message
        raise FileNotFoundError(f"Directory not found: {directory.resolve()}")

    mat_files = sorted(directory.glob("*.mat"))
    if not mat_files:
        # Use the resolved absolute path in the error message
        raise FileNotFoundError(f"No .mat files found in {directory.resolve()}")

    for mat_file in mat_files:
        try:
            data = scipy.io.loadmat(mat_file)
            for key, val in data.items():
                if isinstance(val, np.ndarray):
                    swings.append(val)
                    break
        except Exception as e:
            print(f"Error loading {mat_file.name}: {str(e)}")

    if not swings:
        raise ValueError(f"No valid pose data loaded from {directory}")
    return swings


# Load datasets with verification
try:
    print(f"Loading good swings from {GOOD_DIR}...")
    good_swings = load_swings(GOOD_DIR)
    print(f"Loaded {len(good_swings)} good swings")

    print(f"\nLoading bad swings from {BAD_DIR}...")
    bad_swings = load_swings(BAD_DIR)
    print(f"Loaded {len(bad_swings)} bad swings")
except Exception as e:
    print(f"\nFatal Error: {str(e)}")
    exit(1)


def generate_shuffled_swings(num_samples=120):
    """Generate hybrid swings, saving pose to .mat and metrics to .csv"""
    if not good_swings or not bad_swings:
        raise ValueError("No input data available for shuffling")

    for i in tqdm(range(1, num_samples + 1)):
        base_idx = random.randint(0, len(good_swings) - 1)
        donor_idx = random.randint(0, len(bad_swings) - 1)

        mixed_swing = np.copy(good_swings[base_idx])
        metrics = np.ones(9)  # Initialize metrics array

        for metric_idx in range(9):
            if random.random() > 0.5:
                mixed_swing[:, landmark_group[metric_idx], :] = \
                    bad_swings[donor_idx][:, landmark_group[metric_idx], :]
                metrics[metric_idx] = 0  # Update metrics based on shuffling

        # Define base filename
        base_filename = f"shuf_swing_{i:03d}"
        mat_filepath = OUTPUT_DIR / f"{base_filename}.mat"
        csv_filepath = OUTPUT_DIR / f"{base_filename}.csv"

        # Save only the pose data to the .mat file
        # Assuming the original files also have a single key, we use 'pose'
        scipy.io.savemat(
            mat_filepath,
            {
                'pose': mixed_swing
            }
            # Consider adding do_compression=True if originals are compressed
        )

        # Save the metrics array to a .csv file
        # Use fmt='%d' to save as integers (0 or 1)
        np.savetxt(csv_filepath, metrics, delimiter=",", fmt='%d')

        # Update print statement to reflect both files saved
        print(f"Generated {base_filename}.mat and {base_filename}.csv", end='\r')
    print("\n") # Add a newline after the loop finishes


# Generate dataset
try:
    print("\nGenerating shuffled swings...")
    generate_shuffled_swings()
    print("\nGeneration completed successfully!")

    # Verification (Optional: Update or remove as needed)
    # This verification only checks the .mat file
    sample_mat_file = next(OUTPUT_DIR.glob("shuf_swing_*.mat"), None)
    if sample_mat_file:
        data = scipy.io.loadmat(sample_mat_file)
        print("\nSample .mat file verification:")
        print(f"File: {sample_mat_file.name}")
        if 'pose' in data:
             print(f"Pose shape: {data['pose'].shape}")
        else:
             print(f"Key 'pose' not found in {sample_mat_file.name}")
        # Add verification for the corresponding .csv file if desired
        sample_csv_file = sample_mat_file.with_suffix('.csv')
        if sample_csv_file.exists():
            print(f"Corresponding CSV file found: {sample_csv_file.name}")
            try:
                loaded_metrics = np.loadtxt(sample_csv_file, delimiter=',', dtype=int)
                print(f"Metrics from CSV: {loaded_metrics}")
            except Exception as e:
                print(f"Error loading CSV {sample_csv_file.name}: {e}")
        else:
            print(f"Warning: Corresponding CSV file not found for {sample_mat_file.name}")

    else:
        print("Warning: No output .mat files found for verification")
except Exception as e:
    print(f"\nError during generation: {str(e)}")

# Remove or comment out the specific file loading test section if not needed
# mat_file_path = BASE_DIR / 'Good/Good Iron Swing - Back 1.mat'
# try:
#     data = scipy.io.loadmat(mat_file_path)
#     print("Successfully loaded:", mat_file_path)
#     print("\nKeys in the .mat file:")
#     print(data.keys())
# 
#     print("\nChecking shapes of numpy arrays:")
#     for key, val in data.items():
#         if isinstance(val, np.ndarray):
#             print(f"  Key: '{key}', Shape: {val.shape}, Data Type: {val.dtype}")
#         else:
#             print(f"  Key: '{key}', Type: {type(val)}")
# 
# except FileNotFoundError:
#     print(f"Error: File not found at {mat_file_path}")
# except Exception as e:
#     print(f"Error loading {mat_file_path}: {e}")