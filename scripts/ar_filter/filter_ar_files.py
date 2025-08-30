import xarray as xr
import pandas as pd

# --- Configuration ---
CATALOG_FILE = 'globalARcatalog_ERA5_1940-2024_v4.0.nc'
OUTPUT_FILE = 'ar_dates_bangladesh_2010-2023.txt'

# 1. Define our geographical and temporal bounds
# Bounding box for Bangladesh and the Bay of Bengal
LAT_MIN, LAT_MAX = 20, 27
LON_MIN, LON_MAX = 88, 93
# Time period of interest
START_DATE, END_DATE = '2010-01-01', '2023-12-31'

print("Starting AR date filtering process...")

# Open the dataset (xarray uses lazy loading, no data is loaded yet)
ds = xr.open_dataset(CATALOG_FILE)

# 2. Select the data for our specific region and time period
# Use .sel() which works with coordinate labels. This is fast and efficient.
print(f"Slicing dataset for {START_DATE} to {END_DATE} and the region...")
regional_ds = ds.sel(
    time=slice(START_DATE, END_DATE),
    lat=slice(LAT_MAX, LAT_MIN), # Note: latitude is in descending order
    lon=slice(LON_MIN, LON_MAX)
)

# 3. Find all time steps where an AR is present in the box
# We check the 'shapemap'. If the sum of the mask over lat/lon is > 0,
# it means at least one grid cell in our box has an AR.
print("Identifying time steps with AR presence...")
# This operation is the most intensive part. Xarray reads the data it needs from disk.
ar_present_times = regional_ds['shapemap'].sum(dim=['lat', 'lon', 'ens', 'lev']) > 0

# Extract the actual time coordinates where ARs were present
event_dates = regional_ds['time'].where(ar_present_times, drop=True)

# 4. Save the list of dates to a text file
print(f"Found {len(event_dates)} 6-hourly time steps with ARs.")
# Convert to a pandas DatetimeIndex for easy formatting and saving
event_dates_pd = event_dates.to_series()
event_dates_pd.to_csv(OUTPUT_FILE, index=False, header=False, date_format='%Y-%m-%dT%H:%M:%S')

print(f"\n✅ Successfully saved event dates to {OUTPUT_FILE}")