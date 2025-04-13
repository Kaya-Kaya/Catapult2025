import os
import scipy.io
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

def rename_mat_keys_to_pose():
    """Renames the main variable key inside data_*.mat files to 'pose'."""
    try:
        script_dir = Path(__file__).parent.resolve()
        base_dir = script_dir.parent # Workspace root
        data_dir = base_dir / "data"

        if not data_dir.exists():
            print(f"Error: Data directory not found at {data_dir}", file=sys.stderr)
            return

        print(f"Scanning directory: {data_dir}")
        mat_files = sorted(list(data_dir.glob("data_*.mat")))

        if not mat_files:
            print(f"Warning: No 'data_*.mat' files found in {data_dir}. Nothing to do.", file=sys.stderr)
            return

        print(f"Found {len(mat_files)} data files. Processing...")
        files_changed = 0
        files_skipped = 0
        files_error = 0

        for mat_path in tqdm(mat_files, desc="Renaming keys"):
            try:
                mat_data = scipy.io.loadmat(mat_path)

                original_key = None
                pose_data = None

                # Find the first non-internal key holding a numpy array
                for key, value in mat_data.items():
                    if not key.startswith('__') and isinstance(value, np.ndarray):
                        original_key = key
                        pose_data = value
                        break # Assume only one main data array per file

                if original_key is None or pose_data is None:
                    print(f"\nWarning: No suitable data array found in {mat_path.name}. Skipping.", file=sys.stderr)
                    files_skipped += 1
                    continue

                if original_key == 'pose':
                    # print(f"\nInfo: Key in {mat_path.name} is already 'pose'. Skipping.") # Optional info
                    files_skipped += 1
                    continue

                # Key needs renaming, create new dict and save
                new_data = {'pose': pose_data}
                scipy.io.savemat(mat_path, new_data, do_compression=True)
                files_changed += 1
                # print(f"\nInfo: Renamed key in {mat_path.name} from '{original_key}' to 'pose'.") # Optional info

            except Exception as e:
                print(f"\nError processing file {mat_path.name}: {e}", file=sys.stderr)
                files_error += 1

        print("\nKey renaming process complete.")
        print(f"Files changed: {files_changed}")
        print(f"Files skipped (already 'pose' or no data found): {files_skipped}")
        print(f"Files with errors: {files_error}")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    rename_mat_keys_to_pose() 