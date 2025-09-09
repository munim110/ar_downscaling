import pandas as pd
import os
import sys
import requests
import bz2
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# --- CONFIGURATION ---
# Corrected dates based on your findings
H8_MULTI_SEGMENT_START_DATE = pd.to_datetime('2019-12-09T18:10:00')
H9_START_DATE = pd.to_datetime('2022-12-01')

HISD2NETCDF_EXE = 'hisd2netcdf/hisd2netcdf' 
# Available at: https://atmos.washington.edu/~brodzik/public/miked/precip/Himawari/hisd2netcdf/
# Make sure to follow the installation instructions there.

DATE_FILE = 'ar_dates_bangladesh_2015-2023.txt'
BASE_OUTPUT_DIR = Path('himawari')
BAND = 8 # Water Vapor
RESOLUTION_CODE = 20 # 2km resolution code for H8/H9
RESOLUTION_DEG = 0.02 # Output resolution in degrees

# Bounding box for Bangladesh
LAT_TOP, LAT_BOTTOM = 27, 20
LON_LEFT, LON_RIGHT = 88, 93

# --- HELPER FUNCTION ---
def download_single_segment(url: str, output_path: Path):
    try:
        if output_path.exists(): return True
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        output_path.write_bytes(bz2.decompress(response.content))
        return True
    except Exception as e:
        print(f"Failed segment download for {url}: {e}", file=sys.stderr)
        return False

# --- UNIFIED WORKFLOW FUNCTION ---
def process_timestamp(dt: datetime):
    # Determine satellite, segment count, and naming conventions based on the date
    if dt < H8_MULTI_SEGMENT_START_DATE:
        satellite, sat_prefix, bucket = ("himawari8", "HS_H08", "noaa-himawari8")
        num_segments, seg_format = (1, "{s:02d}01")
    elif dt < H9_START_DATE:
        satellite, sat_prefix, bucket = ("himawari8", "HS_H08", "noaa-himawari8")
        num_segments, seg_format = (10, "{s:02d}10")
    else:
        satellite, sat_prefix, bucket = ("himawari9", "HS_H09", "noaa-himawari9")
        num_segments, seg_format = (10, "{s:02d}10")
    
    print(f"\n--- Processing {satellite.upper()} ({num_segments} segments) for {dt} ---")
    
    # Set up paths
    final_nc_path = BASE_OUTPUT_DIR / f"{sat_prefix}_{dt.strftime('%Y%m%d_%H%M')}_B{BAND:02d}_BANGLADESH.nc"
    if final_nc_path.exists():
        print(f"Final file exists, skipping: {final_nc_path.name}")
        return

    temp_dat_dir = BASE_OUTPUT_DIR / 'temp_dat' / dt.strftime('%Y%m%d_%H%M')
    temp_dat_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"https://{bucket}.s3.amazonaws.com/AHI-L1b-FLDK/{dt.strftime('%Y/%m/%d/%H%M')}"

    # Download all segments
    print(f"Downloading {num_segments} segment(s)...")
    input_files_to_convert = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for s in range(1, num_segments + 1):
            segment_str = seg_format.format(s=s)
            filename = f"{sat_prefix}_{dt.strftime('%Y%m%d_%H%M')}_B{BAND:02d}_FLDK_R{RESOLUTION_CODE:02d}_S{segment_str}.DAT.bz2"
            url = f"{base_url}/{filename}"
            output_path = temp_dat_dir / filename[:-4]
            input_files_to_convert.append(output_path)
            futures.append(executor.submit(download_single_segment, url, output_path))
    
    if not all(f.result() for f in futures):
        print(f"ERROR: Failed to download all segments for {dt}. Skipping.", file=sys.stderr)
        return

    # Convert segments to a single subsetted NetCDF
    print(f"Converting segments for {dt}...")
    width = int((LON_RIGHT - LON_LEFT) / RESOLUTION_DEG) + 1
    height = int((LAT_TOP - LAT_BOTTOM) / RESOLUTION_DEG) + 1
    cmd = [HISD2NETCDF_EXE, '-width', str(width), '-height', str(height), '-lat', str(LAT_TOP), '-lon', str(LON_LEFT), '-dlat', str(RESOLUTION_DEG), '-dlon', str(RESOLUTION_DEG), '-o', str(final_nc_path)]
    for f in input_files_to_convert: cmd.extend(['-i', str(f)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR during hisd2netcdf conversion for {dt}:\n{result.stderr}", file=sys.stderr)
    else:
        print(f"SUCCESS: Created {final_nc_path.name}")

    # Clean up temporary files
    print("Cleaning up temporary segment files...")
    for f in input_files_to_convert:
        try: f.unlink()
        except OSError as e: print(f"Error during cleanup: {e}", file=sys.stderr)

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    if not os.path.exists(HISD2NETCDF_EXE):
        print(f"ERROR: Conversion tool not found at '{HISD2NETCDF_EXE}'", file=sys.stderr)
        sys.exit(1)
        
    BASE_OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Run in full mode on all dates from the file
    event_dates = pd.to_datetime(pd.read_csv(DATE_FILE, header=None)[0])
    
    for date in event_dates:
        process_timestamp(date)

    print("\n\n✅ Unified workflow complete.")
