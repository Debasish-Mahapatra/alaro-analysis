import xarray as xr
import faxarray
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import gc

# --- Configuration ---
DATA_ROOT = "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/control/untar-output"
OUTPUT_DIR = "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "hydrometeors_profile.png")
MAX_CORES = 20

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
YEARS = ["2014", "2015"]


def process_file_path(file_path):
    """
    Worker function to process a single FA file.
    Opens lazily, crops, computes mean, and returns small numpy arrays.
    """
    try:
        # Open the dataset (lazy by default)
        ds = xr.open_dataset(file_path, engine="faxarray", variables=VARIABLES)
        
        if 'lat' in ds.coords and 'lon' in ds.coords:
            lat_c, lon_c = 'lat', 'lon'
        elif 'latitude' in ds.coords and 'longitude' in ds.coords:
            lat_c, lon_c = 'latitude', 'longitude'
        else:
            ds.close()
            return file_path, None

        # Create mask
        mask = (
            (ds[lon_c] >= LON_MIN) & (ds[lon_c] <= LON_MAX) &
            (ds[lat_c] >= LAT_MIN) & (ds[lat_c] <= LAT_MAX)
        )
        
        # Apply mask lazily
        roi = ds.where(mask, drop=True)
        spatial_dims = [d for d in roi.dims if d not in ['level', 'time']]
        
        # Compute mean profile (this is the only step that reads data into memory)
        mean_profile = roi.mean(dim=spatial_dims).compute()
        
        # Geopotential convention
        if "GEOPOTENTIEL" in mean_profile:
             mean_profile["GEOPOTENTIEL_HEIGHT"] = mean_profile["GEOPOTENTIEL"] / 9.80665

        # Extract underlying numpy arrays to avoid serializing entire xarray objects back
        profile_dict = {}
        for var in VARIABLES + ["GEOPOTENTIEL_HEIGHT"]:
            if var in mean_profile:
                profile_dict[var] = mean_profile[var].values.copy()
                
        # Close dataset explicitly to free file handles and resources
        ds.close()
        
        # Force garbage collection to keep worker memory clean
        del ds, mask, roi, mean_profile
        gc.collect()

        return file_path, profile_dict

    except Exception as e:
        return file_path, None

def main():
    print(f"Starting Hydrometeor Profile Analysis using up to {MAX_CORES} cores...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Gather file paths
    all_files = []
    for year in YEARS:
        year_pattern = os.path.join(DATA_ROOT, f"pf{year}*")
        for day_dir in sorted(glob.glob(year_pattern)):
            if not os.path.isdir(day_dir):
                continue
            for hour in range(1, 25): 
                filename = f"pfABOFABOF+{hour:04d}"
                file_path = os.path.join(day_dir, filename)
                if os.path.exists(file_path):
                    all_files.append(file_path)
                    
    total_files = len(all_files)
    print(f"Discovered {total_files} files to process.")
    
    if total_files == 0:
        return
        
    accumulated_sums = {
        "LIQUID_WATER": None,
        "SOLID_WATER": None,
        "GRAUPEL": None,
        "SNOW": None,
        "RAIN": None,
        "GEOPOTENTIEL_HEIGHT": None
    }
    
    processed_count = 0
    failed_count = 0

    # 2. Process in parallel using ProcessPoolExecutor
    print("Submitting to ProcessPool...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_CORES) as executor:
        # submit all jobs
        futures = {executor.submit(process_file_path, fp): fp for fp in all_files}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            file_path, profile = future.result()
            
            if profile is not None:
                # Accumulate globally
                for var in accumulated_sums.keys():
                    if var in profile:
                        if accumulated_sums[var] is None:
                            accumulated_sums[var] = profile[var].copy()
                        else:
                            accumulated_sums[var] += profile[var]
                processed_count += 1
            else:
                failed_count += 1
                
            # Logging
            if i % 50 == 0 or i == total_files:
                print(f" Progress: [ {i:5d} / {total_files} ] | Processed: {processed_count} | Failed: {failed_count}")

    if processed_count == 0:
        print("No files were successfully processed.")
        return

    print(f"\nCompleted! Successfully processed {processed_count} files.")
    
    # 3. Compute final means
    final_means = {}
    for var, total_sum in accumulated_sums.items():
        if total_sum is not None:
             final_means[var] = total_sum / processed_count
             
    # Extract height data (y-axis)
    if "GEOPOTENTIEL_HEIGHT" not in final_means:
        y_data = np.arange(1, len(final_means.get("LIQUID_WATER", [])) + 1)
        y_label = "Model Level"
        invert_y = True
    else:
        y_data = final_means["GEOPOTENTIEL_HEIGHT"]
        if y_data.ndim > 1:
            y_data = y_data.ravel()
        y_label = "Geopotential Height (m)"
        invert_y = False

    # 4. Plotting
    print(f"Generating plot at {OUTPUT_FILE}...")
    plt.figure(figsize=(10, 8))
    
    colors = {
        "LIQUID_WATER": "blue",
        "SOLID_WATER": "cyan",
        "GRAUPEL": "red",
        "SNOW": "magenta",
        "RAIN": "green"
    }
    
    for var in ["LIQUID_WATER", "SOLID_WATER", "GRAUPEL", "SNOW", "RAIN"]:
        if var in final_means and final_means[var] is not None:
            v_data = final_means[var]
            if v_data.ndim > 1:
                v_data = v_data.ravel()
            plt.plot(v_data, y_data, label=var.replace("_", " ").title(), color=colors[var], linewidth=2)
            
    if invert_y:
        plt.gca().invert_yaxis()
        
    plt.xlabel("Concentration / Mixing Ratio (kg/kg)", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title("Vertical Profile of Hydrometeors\\n(2-yr Mean, Region: 4°N to 10°S, 67°W to 53°W)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=350)
    print("Done!")

if __name__ == "__main__":
    main()
