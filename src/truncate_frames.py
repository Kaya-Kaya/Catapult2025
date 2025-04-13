import os
import numpy as np
import scipy.io
import glob
import sys

def find_min_frames(folders):
    min_frames = float('inf')
    mat_files = []
    for folder in folders:
        current_files = glob.glob(os.path.join(folder, '*.mat'))
        if not current_files:
            print(f"Warning: No .mat files found in {folder}", file=sys.stderr)
            continue
        mat_files.extend(current_files)
        for file_path in current_files:
            try:
                mat_data = scipy.io.loadmat(file_path)
                # Assuming the relevant data is the first variable with >= 3 dimensions
                for key, value in mat_data.items():
                    if isinstance(value, np.ndarray) and value.ndim >= 3:
                        frames = value.shape[0]
                        min_frames = min(min_frames, frames)
                        # print(f"File: {os.path.basename(file_path)}, Frames: {frames}") # Debugging
                        break # Assume first suitable array is the target
                else:
                    print(f"Warning: No suitable data array found in {file_path}", file=sys.stderr)
            except Exception as e:
                print(f"Error loading {file_path}: {e}", file=sys.stderr)

    if not mat_files:
        print("Error: No .mat files found in any specified folder.", file=sys.stderr)
        return None, []
    if min_frames == float('inf'):
        print("Error: Could not determine frame count from any .mat file.", file=sys.stderr)
        return None, []

    # print(f"Minimum frames found: {min_frames}") # Debugging
    return min_frames, mat_files

def truncate_mat_files(min_frames, file_paths):
    if min_frames is None:
        print("Aborting truncation due to previous errors.", file=sys.stderr)
        return

    print(f"Truncating all files to {min_frames} frames...")
    for file_path in file_paths:
        try:
            mat_data = scipy.io.loadmat(file_path)
            updated = False
            data_to_save = {}
            # Find the key of the data array again and truncate
            for key, value in mat_data.items():
                if isinstance(value, np.ndarray) and value.ndim >= 3 and value.shape[0] >= min_frames:
                    # print(f"Truncating {os.path.basename(file_path)} ({value.shape[0]} -> {min_frames} frames)") # Debugging
                    data_to_save[key] = value[:min_frames, ...]
                    updated = True
                else:
                    # Copy other variables as is
                    data_to_save[key] = value

            if updated:
                scipy.io.savemat(file_path, data_to_save, do_compression=True)
            else:
                 print(f"Warning: Did not find data to truncate in {file_path} or already short enough.", file=sys.stderr)

        except Exception as e:
            print(f"Error processing or saving {file_path}: {e}", file=sys.stderr)

def main():
    base_pose_folder = '../Poses' # Assuming script is run from workspace root
    good_folder = os.path.join(base_pose_folder, 'Good')
    bad_folder = os.path.join(base_pose_folder, 'Bad')

    if not os.path.isdir(good_folder):
        print(f"Error: Directory not found: {good_folder}", file=sys.stderr)
        return
    if not os.path.isdir(bad_folder):
        print(f"Error: Directory not found: {bad_folder}", file=sys.stderr)
        return

    min_frames, mat_files = find_min_frames([good_folder, bad_folder])

    if min_frames is not None:
        truncate_mat_files(min_frames, mat_files)
        print("Truncation complete.")
    else:
        print("Truncation failed.")

if __name__ == "__main__":
    main() 