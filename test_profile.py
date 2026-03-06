import xarray as xr
import faxarray
import sys

def main():
    file_path = "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/control/untar-output/pf20140101/pfABOFABOF+0001"
    print(f"Testing on file: {file_path}")
    
    try:
        ds = xr.open_dataset(file_path, engine="faxarray")
        
        print(f"\\nCoordinates:")
        print(ds.coords)
        
        # Test required variables
        required = ["LIQUID_WATER", "SOLID_WATER", "GRAUPEL", "SNOW", "RAIN", "GEOPOTENTIEL"]
        available = list(ds.data_vars.keys())
        
        print(f"\\nVerifying required variables:")
        all_found = True
        for req in required:
            if req in available:
                print(f"  [OK] {req} found, shape={ds[req].shape}")
            else:
                print(f"  [MISSING] {req} not found")
                all_found = False
                
        if not all_found:
            print("To see all avilable: ", available[:10], "...")
            
    except Exception as e:
        print(f"Error opening/reading dataset: {e}")

if __name__ == "__main__":
    main()
