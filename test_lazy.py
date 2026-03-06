import xarray as xr
import faxarray
import os
import glob
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_ROOT = "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/control/untar-output"
OUTPUT_DIR = "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "hydrometeors_profile_lazy_test.png")

VARIABLES = [
    "LIQUID_WATER",
    "SOLID_WATER",
    "GRAUPEL",
    "SNOW",
    "RAIN",
    "GEOPOTENTIEL"
]

LAT_MIN, LAT_MAX = -10.0, 4.0
LON_MIN, LON_MAX = -67.0, -53.0

def main():
    print("Testing Hydrometeor Profile Analysis (Serial Lazy Loading) on 1 Day...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_files = []
    day_dir = os.path.join(DATA_ROOT, "pf20140101")
    for hour in range(1, 25): 
        filename = f"pfABOFABOF+{hour:04d}"
        file_path = os.path.join(day_dir, filename)
        if os.path.exists(file_path):
            all_files.append(file_path)
                    
    print(f"Discovered {len(all_files)} files. Building lazy dataset serially...")

    try:
        # Opening mfdataset with parallel=False. 
        # This builds the dask graph sequentially, avoiding any Fortran multithreading contention at read time.
        ds = xr.open_mfdataset(
            all_files, 
            engine="faxarray", 
            parallel=False,    # <--- Critical for thread-safety in backend readers       
            combine="nested",        
            concat_dim="time",       
            data_vars=VARIABLES,
            coords="minimal",
            compat="override"        
        )
        
        if 'lat' in ds.coords and 'lon' in ds.coords:
            lat_coord, lon_coord = 'lat', 'lon'
        elif 'latitude' in ds.coords and 'longitude' in ds.coords:
            lat_coord, lon_coord = 'latitude', 'longitude'
        else:
            print("Could not find standard latitude/longitude coordinates.")
            return

        print("Dataset opened lazily. Setting up compute graph...")

        mask = (
            (ds[lon_coord] >= LON_MIN) & (ds[lon_coord] <= LON_MAX) &
            (ds[lat_coord] >= LAT_MIN) & (ds[lat_coord] <= LAT_MAX)
        )
        roi = ds.where(mask, drop=True)

        spatial_dims = [d for d in roi.dims if d not in ['level', 'time']]
        
        print("Defining temporal and spatial means...")
        mean_profile = roi.mean(dim=['time'] + spatial_dims)
        
        if "GEOPOTENTIEL" in mean_profile:
             mean_profile["GEOPOTENTIEL_HEIGHT"] = mean_profile["GEOPOTENTIEL"] / 9.80665

        # Print the dask array specs before executing
        print(mean_profile)

        print("Triggering compute...")
        # Since reading the FA files is done via chunks now, we calculate sequentially
        import dask
        with dask.config.set(scheduler='synchronous'):
             computed_profile = mean_profile.compute()
            
        print("Compute finished!")
        print(computed_profile)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Fatal error during execution: {e}")

if __name__ == "__main__":
    main()
