import pandas as pd
from pathlib import Path
import re
from tqdm import tqdm

# --- Configuration ---
# Ensure these paths are correct for your environment
ERA5_DIR = Path('./era5_data_new')
SATELLITE_DIR = Path('./satellite_data')
OUTPUT_MANIFEST = 'data_manifest_combined.csv'

# --- Script ---
print("Scanning data directories for combined ERA5 files...")

# --- Scan ERA5 directory for the new combined files ---
era5_files = {}
# --- The script looks for the '_era5_combined.nc' files ---
for f in ERA5_DIR.glob('*_era5_combined.nc'):
    # Filename format is still: YYYY-MM_...
    match = re.search(r'(\d{4})-(\d{2})', f.name)
    if match:
        year, month = map(int, match.groups())
        era5_files[(year, month)] = f

if not era5_files:
    print("❌ Error: No combined ERA5 files found. Make sure the download was successful and the ERA5_DIR path is correct.")
else:
    print(f"✅ Found {len(era5_files)} combined ERA5 NetCDF files.")

    # --- Scan Satellite directory ---
    satellite_pairs = {}
    print("Scanning satellite data directory...")
    for f in tqdm(SATELLITE_DIR.glob('*.nc'), desc="Scanning satellite files"):
        # Filename format: HS_H0X_YYYYMMDD_HHMM_...
        match = re.search(r'_(\d{8})_(\d{4})_', f.name)
        if match:
            date_str, time_str = match.groups()
            timestamp = pd.to_datetime(f"{date_str}{time_str}", format='%Y%m%d%H%M')
            satellite_pairs[timestamp] = f
    
    if not satellite_pairs:
        print("❌ Error: No satellite files found. Check the SATELLITE_DIR path.")
    else:
        print(f"✅ Found {len(satellite_pairs)} satellite files.")

        # --- Create the manifest by matching satellite timestamps to monthly ERA5 files ---
        print("Pairing satellite and ERA5 data...")
        matched_data = []
        for ts, sat_file in tqdm(satellite_pairs.items(), desc="Creating manifest"):
            year_month_key = (ts.year, ts.month)
            if year_month_key in era5_files:
                era5_file = era5_files[year_month_key]
                matched_data.append({
                    'timestamp': ts,
                    'satellite_path': sat_file,
                    'era5_path': era5_file
                })

        if not matched_data:
            print("❌ Error: No matched data pairs found. Check if the date ranges of your ERA5 and satellite data overlap.")
        else:
            # Create a pandas DataFrame
            manifest_df = pd.DataFrame(matched_data)
            manifest_df = manifest_df.set_index('timestamp').sort_index()

            # Save to a CSV file
            manifest_df.to_csv(OUTPUT_MANIFEST)
            
            print(f"\n🎉 Successfully created data manifest!")
            print(f"   Total matched pairs: {len(manifest_df)}")
            print(f"   Manifest file saved to: {OUTPUT_MANIFEST}")
            print("\n--- Sample of Manifest ---")
            print(manifest_df.head())
