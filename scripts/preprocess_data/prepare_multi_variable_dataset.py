import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
import joblib
import shutil

# --- Configuration ---
# Directory where your multi-variable .npy files are located
PROCESSED_DIR = Path('./data_processed_multi_variable')
# Directory where the final train/val/test splits will be created
FINAL_DATA_DIR = Path('./final_dataset_multi_variable')
# The manifest file created in the previous step
MANIFEST_FILE = 'data_manifest_combined.csv'
# The output file for our new, multi-channel normalization stats
STATS_FILE = FINAL_DATA_DIR / 'normalization_stats_multi_variable.joblib'

# Chronological split percentages
VAL_SIZE = 0.10  # 10% for validation
TEST_SIZE = 0.10 # 10% for testing
# Training size will be the remaining 80%

# --- Main Script ---
if __name__ == "__main__":
    # 1. Create final directories
    FINAL_DATA_DIR.mkdir(exist_ok=True)
    (FINAL_DATA_DIR / 'train').mkdir(exist_ok=True)
    (FINAL_DATA_DIR / 'val').mkdir(exist_ok=True)
    (FINAL_DATA_DIR / 'test').mkdir(exist_ok=True)

    # 2. Load manifest and create chronological splits
    manifest_df = pd.read_csv(MANIFEST_FILE, index_col='timestamp', parse_dates=True).sort_index()
    
    n_samples = len(manifest_df)
    n_test = int(n_samples * TEST_SIZE)
    n_val = int(n_samples * VAL_SIZE)
    
    test_manifest = manifest_df.iloc[-n_test:]
    val_manifest = manifest_df.iloc[-(n_test + n_val):-n_test]
    train_manifest = manifest_df.iloc[:-(n_test + n_val)]
    
    print(f"Dataset split:")
    print(f"  Training samples:   {len(train_manifest)}")
    print(f"  Validation samples: {len(val_manifest)}")
    print(f"  Test samples:       {len(test_manifest)}")

    # 3. Calculate per-channel normalization statistics ONLY from the training set
    print("\nCalculating per-channel normalization statistics from the training set...")
    
    # Use Welford's algorithm for stable online calculation of mean and variance
    # This is memory-efficient as it doesn't require loading all data at once.
    count = 0
    # Initialize for 5 channels
    mean = np.zeros(5)
    M2 = np.zeros(5)

    for timestamp, row in tqdm(train_manifest.iterrows(), total=len(train_manifest), desc="Calculating Stats"):
        filepath = PROCESSED_DIR / f"{timestamp.strftime('%Y%m%d_%H%M')}_predictor.npy"
        if filepath.exists():
            # data shape is (channels, height, width)
            data = np.load(filepath)
            # Flatten height and width to get (channels, n_pixels)
            data_flat = data.reshape(data.shape[0], -1)
            
            # Update stats for each channel
            for i in range(data_flat.shape[0]): # Loop through channels
                for x in data_flat[i, :]: # Loop through pixels in the channel
                    count += 1
                    delta = x - mean[i]
                    mean[i] += delta / count
                    delta2 = x - mean[i]
                    M2[i] += delta * delta2

    variance = M2 / (count - 1)
    std_dev = np.sqrt(variance)

    # Calculate stats for the target variable as well
    target_values = []
    for timestamp, row in tqdm(train_manifest.iterrows(), total=len(train_manifest), desc="Calculating Target Stats"):
        filepath = PROCESSED_DIR / f"{timestamp.strftime('%Y%m%d_%H%M')}_target.npy"
        if filepath.exists():
            target_values.append(np.load(filepath).flatten())
            
    target_values = np.concatenate(target_values)
    target_mean = np.mean(target_values)
    target_std = np.std(target_values)
            
    stats = {
        'predictor_mean': mean,
        'predictor_std': std_dev,
        'target_mean': target_mean,
        'target_std': target_std,
        'variables': ['ivt', 't_500', 't_850', 'rh_700', 'w_500']
    }
    
    joblib.dump(stats, STATS_FILE)
    print("\n✅ Per-channel normalization stats calculated and saved:")
    for i, var in enumerate(stats['variables']):
        print(f"  - {var}: Mean={stats['predictor_mean'][i]:.4f}, Std={stats['predictor_std'][i]:.4f}")
    print(f"  - Target TBB: Mean={stats['target_mean']:.4f}, Std={stats['target_std']:.4f}")

    # 4. Copy files to their final train/val/test directories
    print("\nCopying files to train/val/test directories...")
    
    sets_to_process = {
        'train': train_manifest,
        'val': val_manifest,
        'test': test_manifest
    }
    
    for set_name, manifest in sets_to_process.items():
        print(f"Processing '{set_name}' set...")
        destination_dir = FINAL_DATA_DIR / set_name
        for timestamp, row in tqdm(manifest.iterrows(), total=len(manifest)):
            base_savename = timestamp.strftime('%Y%m%d_%H%M')
            
            # Source files
            predictor_source = PROCESSED_DIR / f"{base_savename}_predictor.npy"
            target_source = PROCESSED_DIR / f"{base_savename}_target.npy"
            
            # Destination files
            predictor_dest = destination_dir / f"{base_savename}_predictor.npy"
            target_dest = destination_dir / f"{base_savename}_target.npy"
            
            # Copy files
            if predictor_source.exists():
                shutil.copy(predictor_source, predictor_dest)
            if target_source.exists():
                shutil.copy(target_source, target_dest)

    print("\n\n✅ Final multi-variable dataset is ready for model training!")
