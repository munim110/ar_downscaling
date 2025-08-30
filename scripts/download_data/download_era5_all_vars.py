import cdsapi
from collections import defaultdict
import pandas as pd
import os
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- Configuration ---
DATE_FILE = 'ar_dates_bangladesh/ar_dates_bangladesh_2015-2023.txt'
OUTPUT_DIR = 'era5_data_new'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Bounding box for Bangladesh and the Bay of Bengal
AREA = [27, 88, 20, 93] # Format: [North, West, South, East]
# Set the maximum number of parallel workers
MAX_WORKERS = 4 # Adjust based on your machine's cores and network speed

def process_month(year, month, days):
    """
    Encapsulates the logic to download, merge, and clean data for a single month.
    This function is designed to be run in a separate process.
    """
    # Initialize the CDS API client within the worker process
    c = cdsapi.Client()
    
    # Define final combined file and temporary component files
    target_file_combined = f"{OUTPUT_DIR}/{year}-{month:02d}_era5_combined.nc"
    temp_file_ivt = f"{OUTPUT_DIR}/{year}-{month:02d}_era5_ivt.tmp.nc"
    temp_file_pl = f"{OUTPUT_DIR}/{year}-{month:02d}_era5_pressure_levels.tmp.nc"

    # If the final combined file already exists, skip
    if os.path.exists(target_file_combined):
        return f"SKIPPED: {os.path.basename(target_file_combined)} already exists."

    try:
        # --- Step A: Download IVT Data (Single Level) ---
        if not os.path.exists(temp_file_ivt):
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis', 'format': 'netcdf',
                    'variable': ['vertical_integral_of_eastward_water_vapour_flux', 'vertical_integral_of_northward_water_vapour_flux'],
                    'year': str(year), 'month': f"{month:02d}", 'day': days,
                    'time': ['00:00', '06:00', '12:00', '18:00'], 'area': AREA,
                },
                temp_file_ivt)
        
        # --- Step B: Download New Physical Variables (Pressure Levels) ---
        if not os.path.exists(temp_file_pl):
            c.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis', 'format': 'netcdf',
                    'variable': ['relative_humidity', 'temperature', 'vertical_velocity'],
                    'pressure_level': ['500', '700', '850'],
                    'year': str(year), 'month': f"{month:02d}", 'day': days,
                    'time': ['00:00', '06:00', '12:00', '18:00'], 'area': AREA,
                },
                temp_file_pl)

        # --- Step C: Merge the two downloaded files ---
        with xr.open_dataset(temp_file_ivt) as ds_ivt, xr.open_dataset(temp_file_pl) as ds_pl:
            merged_ds = xr.merge([ds_ivt, ds_pl])
            merged_ds.to_netcdf(target_file_combined)
        
        return f"SUCCESS: Created {os.path.basename(target_file_combined)}."

    except Exception as e:
        return f"ERROR processing {year}-{month:02d}: {e}"

    finally:
        # --- Step D: Clean up temporary files ---
        if os.path.exists(temp_file_ivt):
            os.remove(temp_file_ivt)
        if os.path.exists(temp_file_pl):
            os.remove(temp_file_pl)

# --- Main Script ---
if __name__ == "__main__":
    # 1. Read and process the event dates
    print(f"Reading event dates from {DATE_FILE}...")
    event_dates = pd.to_datetime(pd.read_csv(DATE_FILE, header=None)[0])

    # Group dates by year and month for efficient API requests
    requests_by_month = defaultdict(lambda: defaultdict(list))
    for dt in event_dates:
        requests_by_month[dt.year][dt.month].append(str(dt.day))

    # Create a list of tasks to be processed
    tasks = []
    for year, months in sorted(requests_by_month.items()):
        for month, days in sorted(months.items()):
            unique_days = sorted(list(set(days)))
            tasks.append((year, month, unique_days))

    # 2. Use ProcessPoolExecutor to run tasks in parallel
    print(f"Starting parallel download and merge process using up to {MAX_WORKERS} workers...")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks to the executor
        futures = {executor.submit(process_month, year, month, days): f"{year}-{month:02d}" for year, month, days in tasks}
        
        # Process results as they complete and show progress with tqdm
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing months"):
            result = future.result()
            print(result)

    print("\n✅ All ERA5 data downloaded and combined successfully.")
    print(f"   Output files saved to: {OUTPUT_DIR}")