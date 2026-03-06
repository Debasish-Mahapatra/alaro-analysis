import xarray as xr
import numpy as np
import pandas as pd

# Files
file_ctrl = 'control_updraft.nc'
file_2mom = '2mom_updraft.nc'
file_graupelp = 'graupel_updraft.nc'

def run_analysis(period_name, hours_utc, level_start, level_end):
    print(f"\nCalculating stats for {period_name} (Levels S{level_start:03d}-S{level_end:03d})...")
    
    def get_stats(filename):
        ds = xr.open_dataset(filename)
        intensity = np.abs(ds['UD_OMEGA'])
        extent = ds['UD_MESH_FRAC']
        
        time_vals = pd.to_datetime(ds['time'].values)
        hours = time_vals.hour
        mask_time = np.isin(hours, hours_utc)
        
        # Select time and levels
        int_subset = intensity.isel(time=mask_time, level=slice(level_start, level_end))
        ext_subset = extent.isel(time=mask_time, level=slice(level_start, level_end))
        
        # Conditional Intensity: Mean of intensity WHERE extent > 0
        mask_active = (ext_subset > 0)
        
        # Flatten and select active points
        int_values = int_subset.values.flatten()
        ext_values = ext_subset.values.flatten()
        mask_values = mask_active.values.flatten()
        
        active_int = int_values[mask_values]
        
        if len(active_int) > 0:
            int_mean = np.mean(active_int)
        else:
            int_mean = np.nan
            
        # Extent is domain mean
        ext_mean = np.mean(ext_values)
        
        ds.close()
        return int_mean, ext_mean

    i_ctrl, e_ctrl = get_stats(file_ctrl)
    i_2mom, e_2mom = get_stats(file_2mom)
    i_graupel, e_graupel = get_stats(file_graupelp)

    print("\n--- ANOMALIES (Exp - Control) ---")
    print(f"2mom Intensity:  {(i_2mom - i_ctrl):.4e} ({((i_2mom-i_ctrl)/i_ctrl)*100:.1f}%)")
    print(f"2mom Extent:     {(e_2mom - e_ctrl):.4e} ({((e_2mom-e_ctrl)/e_ctrl)*100:.1f}%)")
    print(f"Graupel Intensity: {(i_graupel - i_ctrl):.4e} ({((i_graupel-i_ctrl)/i_ctrl)*100:.1f}%)")
    print(f"Graupel Extent:    {(e_graupel - e_ctrl):.4e} ({((e_graupel-e_ctrl)/e_ctrl)*100:.1f}%)")

# Morning Low Levels (13-16 UTC, S070-S086)
run_analysis("Morning Low-Level", [13, 14, 15, 16], 70, 87)

# Afternoon Upper Levels (19-22 UTC, S020-S045)
run_analysis("Afternoon Upper-Level", [19, 20, 21, 22], 20, 45)
