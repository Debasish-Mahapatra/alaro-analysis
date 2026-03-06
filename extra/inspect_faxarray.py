
import xarray as xr
import sys
import os

# Define path
data_path = "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/control/untar-output/pf20140101/pfABOFABOF+0000"

print(f"Opening {data_path}...")
try:
    # faxarray should be registered as an engine if installed
    ds = xr.open_dataset(data_path, engine='faxarray')
    
    print("\n--- Dimensions ---")
    print(ds.dims)
    
    print("\n--- Variables ---")
    # Wrap in list to force printing all, or just print keys if too many
    print(list(ds.data_vars.keys()))
    
    # Check for likely vertical coordinates
    print("\n--- Coordinates ---")
    print(ds.coords)

except Exception as e:
    print(f"Error opening dataset: {e}")
    # Fallback import check
    try:
        import faxarray
        print(f"faxarray version: {faxarray.__version__}")
    except ImportError:
        print("faxarray not found in this environment")
