import xarray as xr
import numpy as np
import pandas as pd

# Files
file_ctrl = 'control_updraft.nc'
file_2mom = '2mom_updraft.nc'
file_graupelp = 'graupel_updraft.nc'

def get_morning_stats(filename, name):
    ds = xr.open_dataset(filename)
    # Flux components
    intensity = np.abs(ds['UD_OMEGA'])
    extent = ds['UD_MESH_FRAC']
    
    # Time (UTC)
    # Morning in Manaus (09-12 LT) = 13-16 UTC
    # Low levels: S070-S086
    
    # Select hours 13, 14, 15, 16 UTC
    time_vals = pd.to_datetime(ds['time'].values)
    hours = time_vals.hour
    mask_morning = np.isin(hours, [13, 14, 15, 16])
    
    # Select levels 70-86
    int_morn = intensity.isel(time=mask_morning, level=slice(70, 87))
    ext_morn = extent.isel(time=mask_morning, level=slice(70, 87))
    
    mean_int = np.nanmean(int_morn.values)
    mean_ext = np.nanmean(ext_morn.values)
    
    print(f"--- {name} ---")
    print(f"Mean {name} Intensity: {mean_int:.4e}")
    print(f"Mean {name} Extent:    {mean_ext:.4e}")
    return mean_int, mean_ext

def run_analysis(period_name, hours_utc, level_start, level_end):
    print(f"\nCalculating stats for {period_name} (Levels S{level_start:03d}-S{level_end:03d})...")
    
    def get_stats(filename):
        ds = xr.open_dataset(filename)
        intensity = np.abs(ds['UD_OMEGA'])
        extent = ds['UD_MESH_FRAC']
        
        time_vals = pd.to_datetime(ds['time'].values)
        hours = time_vals.hour
        mask = np.isin(hours, hours_utc)
        
        int_mean = np.nanmean(intensity.isel(time=mask, level=slice(level_start, level_end)).values)
        ext_mean = np.nanmean(extent.isel(time=mask, level=slice(level_start, level_end)).values)
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
# Note: 15-18 LT = 19-22 UTC
run_analysis("Afternoon Upper-Level", [19, 20, 21, 22], 20, 45)
