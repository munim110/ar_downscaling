import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
# Point this to the directory with your processed files
PROCESSED_DIR = Path('./data_processed')
# Pick one of the files you successfully processed to inspect
FILE_TO_INSPECT = 'processed_20231212_1800.nc'

# --- Script ---
filepath = PROCESSED_DIR / FILE_TO_INSPECT

print(f"🔎 Verifying file: {filepath}")

# 1. Load the processed dataset
ds = xr.open_dataset(filepath)

# 2. Inspect the structure and print a summary
print("\n--- Dataset Summary ---")
print(ds)

# 3. Explicitly check that the shapes of the predictor and target match
predictor_shape = ds['predictor_ivt'].shape
target_shape = ds['target_tbb'].shape

print(f"\nPredictor (IVT) shape: {predictor_shape}")
print(f"Target (TBB) shape:    {target_shape}")

if predictor_shape == target_shape:
    print("✅ Shapes match! The data is correctly aligned.")
else:
    print("❌ ERROR: Shapes do not match!")

# 4. Create a side-by-side visualization
print("\nGenerating plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot Predictor: Upsampled ERA5 IVT
ds['predictor_ivt'].plot(ax=axes[0], cmap='viridis')
axes[0].set_title('Predictor: Upsampled IVT (Blurry)')

# Plot Target: High-Resolution Himawari TBB
# Note: For brightness temperature, lower values (colder) mean higher cloud tops.
# We invert the colormap ('plasma_r') to make these clouds appear brighter.
ds['target_tbb'].plot(ax=axes[1], cmap='plasma_r')
axes[1].set_title('Target: Himawari TBB (Sharp)')

fig.suptitle(f'Verification for {ds.attrs["timestamp"]}', fontsize=16)
plt.tight_layout()
plt.savefig('sample.png')

print("\nVerification complete. Close the plot window to exit.")