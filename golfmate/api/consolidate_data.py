import os
import shutil
import numpy as np
import scipy.io
from pathlib import Path
from tqdm import tqdm
import sys

def consolidate_data():
    """Consolidates pose data and creates corresponding metric files."""
    try:
        script_dir = Path(__file__).parent.resolve()
        base_dir = script_dir.parent # Workspace root
        poses_dir = base_dir / "Poses"
        good_dir = poses_dir / "Good"
        bad_dir = poses_dir / "Bad"
        shuffled_dir = poses_dir / "Shuffled"
        output_data_dir = base_dir / "data" # New directory at workspace root

        # Create the output directory
        output_data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created or found output directory: {output_data_dir}")

        file_index = 0
        source_folders = {
            "Good": good_dir,
            "Bad": bad_dir,
            "Shuffled": shuffled_dir
        }

        for folder_type, source_dir in source_folders.items():
            print(f"\nProcessing folder: {source_dir}")
            if not source_dir.exists():
                print(f"Warning: Source directory not found: {source_dir}. Skipping.", file=sys.stderr)
                continue

            mat_files = sorted(list(source_dir.glob("*.mat")))
            if not mat_files:
                print(f"Warning: No .mat files found in {source_dir}. Skipping.", file=sys.stderr)
                continue

            print(f"Found {len(mat_files)} .mat files.")
            for source_mat_path in tqdm(mat_files, desc=f"Processing {folder_type}"):
                # Define target paths
                target_data_filename = f"data_{file_index:03d}.mat"
                target_metric_filename = f"metric_{file_index:03d}.mat"
                target_data_path = output_data_dir / target_data_filename
                target_metric_path = output_data_dir / target_metric_filename

                # 1. Copy the original .mat file
                try:
                    shutil.copy2(source_mat_path, target_data_path) # copy2 preserves metadata
                except Exception as e:
                    print(f"\nError copying {source_mat_path} to {target_data_path}: {e}", file=sys.stderr)
                    continue # Skip to next file

                # 2. Create and save the corresponding metric .mat file
                metric_array = None
                try:
                    if folder_type == "Good":
                        metric_array = np.ones(9, dtype=int)
                    elif folder_type == "Bad":
                        metric_array = np.zeros(9, dtype=int)
                    elif folder_type == "Shuffled":
                        source_csv_path = source_mat_path.with_suffix('.csv')
                        if source_csv_path.exists():
                            # Load metrics from CSV, ensure integer type
                            metric_array = np.loadtxt(source_csv_path, delimiter=',', dtype=int)
                        else:
                            print(f"\nWarning: Corresponding CSV file not found for {source_mat_path.name}. Skipping metric file generation.", file=sys.stderr)
                            # Optionally remove the copied data file if metric is essential
                            # target_data_path.unlink(missing_ok=True)
                            # continue # Or skip incrementing index if metric must exist

                    if metric_array is not None:
                        # Save the metric array to a .mat file with a consistent key
                        scipy.io.savemat(target_metric_path, {'metric': metric_array}, do_compression=True)
                    else:
                         # Handle cases where metric couldn't be determined (e.g., missing CSV for shuffled)
                         print(f"\nInfo: Metric array not generated for {target_data_filename}. Corresponding metric file not saved.")
                         # Decide if file_index should still be incremented if metric failed
                         # If a metric file MUST exist, you might want to skip incrementing:
                         # continue

                except Exception as e:
                    print(f"\nError generating or saving metric for {source_mat_path.name} (Target: {target_metric_filename}): {e}", file=sys.stderr)
                    # Optionally remove the copied data file if metric saving fails
                    # target_data_path.unlink(missing_ok=True)
                    continue # Skip incrementing and move to next source file

                file_index += 1 # Increment index only after successful copy and metric handling (or acceptable failure)

        print(f"\nConsolidation complete. Total files processed: {file_index}")
        print(f"Output files are in: {output_data_dir}")

    except Exception as e:
        print(f"\nAn unexpected error occurred during consolidation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    consolidate_data() 