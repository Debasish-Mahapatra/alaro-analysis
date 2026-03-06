import xarray as xr
import faxarray
import os
import glob
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_ROOT = "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/control/untar-output"
OUTPUT_DIR = "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/figures"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "hydrometeors_profile_lazy.png")

VARIABLES = [
    "LIQUID_WATER",
    "SOLID_WATER",
    "GRAUPEL",
    "SNOW",
    "RAIN",
    "GEOPOTENTIEL"
]

# Region of interest: 4 N, -10 South, -66 west and -54 east
LAT_MIN, LAT_MAX = -10.0, 4.0
LON_MIN, LON_MAX = -67.0, -53.0

YEARS = ["2014", "2015"]

def main():
    print("Starting Hydrometeor Profile Analysis (Lazy Loading)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Gather file paths
    all_files = []
    for year in YEARS:
        year_pattern = os.path.join(DATA_ROOT, f"pf{year}*")
        for day_dir in sorted(glob.glob(year_pattern)):
            if not os.path.isdir(day_dir):
                continue
            for hour in range(1, 25): # Keep +0001 to +0024
                filename = f"pfABOFABOF+{hour:04d}"
                file_path = os.path.join(day_dir, filename)
                if os.path.exists(file_path):
                    all_files.append(file_path)
                    
    print(f"Discovered {len(all_files)} files. Building lazy dataset...")

    if not all_files:
        print("No files found.")
        return

    try:
        # 2. Open all files lazily using dask
        # We specify the variables to only load what's needed
        ds = xr.open_mfdataset(
            all_files, 
            engine="faxarray", 
            parallel=True,           # Uses dask delayed internally to open files in parallel
            combine="nested",        # files are just a sequence in time usually
            concat_dim="time",       # standard dimension to stack on
            data_vars=VARIABLES,
            coords="minimal",
            compat="override"        # assume all coordinates are the same to speed up opening
        )
        
        # 3. Handle coordinate names generically
        if 'lat' in ds.coords and 'lon' in ds.coords:
            lat_coord, lon_coord = 'lat', 'lon'
        elif 'latitude' in ds.coords and 'longitude' in ds.coords:
            lat_coord, lon_coord = 'latitude', 'longitude'
        else:
            print("Could not find standard latitude/longitude coordinates.")
            return

        print("Dataset opened lazily. Setting up compute graph...")

        # 4. Mask the region
        # In xarray, slicing via .where is memory intensive if drops occur across chunks,
        # but indexing is better if it's a regular grid. For an unrotated 2D netCDF grid, .sel is best.
        # But FA files might be 1D or have 2D lat/lon. If 2D lat/lon, we use .where
        
        if ds[lat_coord].ndim == 2:
            mask = (
                (ds[lon_coord] >= LON_MIN) & (ds[lon_coord] <= LON_MAX) &
                (ds[lat_coord] >= LAT_MIN) & (ds[lat_coord] <= LAT_MAX)
            )
            roi = ds.where(mask, drop=True)
        else:
             # If it's 1D, we can use fast slice (assuming it's sorted or monotonically increasing/decreasing)
             # But let's stick to safe .where for arbitrary FA mapped files.
             mask = (
                (ds[lon_coord] >= LON_MIN) & (ds[lon_coord] <= LON_MAX) &
                (ds[lat_coord] >= LAT_MIN) & (ds[lat_coord] <= LAT_MAX)
            )
             roi = ds.where(mask, drop=True)

        spatial_dims = [d for d in roi.dims if d not in ['level', 'time']]
        
        # 5. Define the mathematical operations lazily
        print("Defining temporal and spatial means...")
        # Mean across time and space.
        mean_profile = roi.mean(dim=['time'] + spatial_dims)
        
        if "GEOPOTENTIEL" in mean_profile:
             mean_profile["GEOPOTENTIEL_HEIGHT"] = mean_profile["GEOPOTENTIEL"] / 9.80665

        # 6. TRIGGER COMPUTE
        # This is where Dask goes to work and processes all 17,500 files chunk-by-chunk
        print("Triggering compute (this will take a while)...")
        computed_profile = mean_profile.compute()
        print("Compute finished!")

        # 7. Plot
        print("Generating plot...")
        plt.figure(figsize=(10, 8))
        
        if "GEOPOTENTIEL_HEIGHT" in computed_profile:
            y_data = computed_profile["GEOPOTENTIEL_HEIGHT"].values
            if y_data.ndim > 1:
                y_data = y_data.ravel()
            y_label = "Geopotential Height (m)"
            invert_y = False
        else:
            y_data = computed_profile['level'].values
            y_label = "Model Level"
            invert_y = True

        colors = {
            "LIQUID_WATER": "blue",
            "SOLID_WATER": "cyan",
            "GRAUPEL": "red",
            "SNOW": "magenta",
            "RAIN": "green"
        }
        
        for var in ["LIQUID_WATER", "SOLID_WATER", "GRAUPEL", "SNOW", "RAIN"]:
            if var in computed_profile:
                v_data = computed_profile[var].values
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
        print(f"Plot saved to: {OUTPUT_FILE}")

    except Exception as e:
        print(f"Fatal error during execution: {e}")

if __name__ == "__main__":
    main()
