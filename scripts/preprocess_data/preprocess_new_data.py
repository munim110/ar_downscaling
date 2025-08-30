import pandas as pd
import xarray as xr
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
import traceback

# --- Configuration ---
MANIFEST_FILE = 'data_manifest_combined.csv'
# Create a new directory for the multi-variable processed data
OUTPUT_DIR = Path('./data_processed_multi_variable')
OUTPUT_DIR.mkdir(exist_ok=True)

# Set the maximum number of parallel workers
MAX_WORKERS = 8 # Adjust based on your machine's cores

# --- Main Processing Function ---
def process_pair(row_tuple):
    """
    Processes a single row from the manifest to create a multi-channel predictor
    and a single-channel target.
    """
    # Unpack the row tuple
    timestamp, satellite_path, era5_path = row_tuple
    
    try:
        # Define output filenames for the final .npy files
        predictor_filename = OUTPUT_DIR / f"{timestamp.strftime('%Y%m%d_%H%M')}_predictor.npy"
        target_filename = OUTPUT_DIR / f"{timestamp.strftime('%Y%m%d_%H%M')}_target.npy"
        
        # Skip if both files already exist
        if predictor_filename.exists() and target_filename.exists():
            return f"Skipped: {predictor_filename.name} already exists.", "skipped"

        # Open the NetCDF files
        with xr.open_dataset(era5_path) as ds_era5, xr.open_dataset(satellite_path) as ds_sat:
            
            # --- 1. Select the correct timestep from the monthly ERA5 file ---
            # FIX: The ncdump output shows the time dimension is named 'valid_time'.
            ds_era5_ts = ds_era5.sel(valid_time=timestamp, method='nearest')

            # --- 2. Prepare all input variables ---
            # a) Calculate IVT magnitude
            # FIX: The ncdump output shows the IVT components are named 'viwve' and 'viwvn'.
            ivt = np.sqrt(ds_era5_ts['viwve']**2 + ds_era5_ts['viwvn']**2)
            ivt.name = 'ivt'
            
            # b) Select other variables at their specific pressure levels
            # FIX: The ncdump output shows the pressure dimension is named 'pressure_level'.
            # The variable short names are 't', 'r', and 'w'.
            t_500 = ds_era5_ts['t'].sel(pressure_level=500)
            t_850 = ds_era5_ts['t'].sel(pressure_level=850)
            rh_700 = ds_era5_ts['r'].sel(pressure_level=700)
            w_500 = ds_era5_ts['w'].sel(pressure_level=500)
            
            variables_to_process = [ivt, t_500, t_850, rh_700, w_500]
            
            # --- 3. Regrid all variables to match the satellite grid ---
            regridded_vars = []
            for var in variables_to_process:
                regridded_var = var.interp(
                    latitude=ds_sat.latitude, 
                    longitude=ds_sat.longitude, 
                    method='linear'
                )
                regridded_vars.append(regridded_var.values)

            # --- 4. Stack into a single multi-channel NumPy array ---
            multi_channel_predictor = np.stack(regridded_vars, axis=0).astype(np.float32)
            
            # --- 5. Prepare the target variable ---
            target_tbb = ds_sat['tbb'].values.astype(np.float32)

            # --- 6. Save the final processed arrays as .npy files ---
            np.save(predictor_filename, multi_channel_predictor)
            np.save(target_filename, target_tbb)

        return f"Success: {predictor_filename.name}", "success"

    except Exception:
        # Return the full error traceback if something goes wrong
        error_message = f"ERROR processing {timestamp}:\n{traceback.format_exc()}"
        return error_message, "error"

# --- Main Workflow ---
if __name__ == "__main__":
    print(f"✅ Output directory confirmed at absolute path: {OUTPUT_DIR.resolve()}")

    # Load the manifest file
    manifest_df = pd.read_csv(MANIFEST_FILE, parse_dates=['timestamp'])
    
    # Create a list of tasks (tuples) to be processed
    tasks = [tuple(row) for row in manifest_df.itertuples(index=False, name=None)]

    print(f"Starting multi-variable preprocessing for {len(tasks)} files using up to {MAX_WORKERS} workers...")
    
    success_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_pair, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Preprocessing files"):
            message, status = future.result()
            if status == "error":
                # Print any errors immediately and in full
                print("\n" + "="*80)
                print(message)
                print("="*80 + "\n")
                error_count += 1
            elif status == "success":
                success_count += 1

    print("\n\n✅ Multi-variable preprocessing complete.")
    
    # --- Final Verification Step ---
    print("\n--- Verification ---")
    # Count the number of predictor files actually created
    final_file_count = len(list(OUTPUT_DIR.glob('*_predictor.npy')))
    print(f"Tasks processed with 'Success' status: {success_count}")
    print(f"Tasks processed with 'Error' status:   {error_count}")
    print(f"Total files found in output directory: {final_file_count}")

    if error_count == 0 and final_file_count >= success_count:
        print("🎉 Verification successful! All expected files were created.")
    else:
        print("⚠️ Verification failed. There is a discrepancy between processed tasks and final files.")
        print("Please review the error messages above.")

