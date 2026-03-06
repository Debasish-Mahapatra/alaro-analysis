#!/usr/bin/env python3
"""Check ALL values for the email response."""
import xarray as xr
import numpy as np
import pandas as pd

files = {
    'control': 'control_updraft.nc',
    '2mom': '2mom_updraft.nc',
    'graupel': 'graupel_updraft.nc'
}

def analyze_period(name, hours_utc, level_start, level_end):
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"Hours UTC: {hours_utc}, Levels: {level_start}-{level_end}")
    print('='*60)
    
    results = {}
    for exp, fname in files.items():
        ds = xr.open_dataset(fname)
        omega = np.abs(ds['UD_OMEGA'].values)
        frac = ds['UD_MESH_FRAC'].values
        
        times = pd.to_datetime(ds['time'].values)
        hours = times.hour
        mask_time = np.isin(hours, hours_utc)
        
        omega_sub = omega[mask_time, level_start:level_end, :, :]
        frac_sub = frac[mask_time, level_start:level_end, :, :]
        
        ext_mean = np.nanmean(frac_sub)
        mask_active = (frac_sub > 0)
        int_mean = np.nanmean(omega_sub[mask_active]) if np.sum(mask_active) > 0 else np.nan
        
        results[exp] = {'intensity': int_mean, 'extent': ext_mean}
        ds.close()
    
    for exp in ['control', '2mom', 'graupel']:
        r = results[exp]
        print(f"{exp.upper()}: Intensity={r['intensity']:.3f} Pa/s, Extent={r['extent']:.6f}")
    
    print("\nANOMALIES:")
    ctrl = results['control']
    for exp in ['2mom', 'graupel']:
        r = results[exp]
        di = (r['intensity'] - ctrl['intensity']) / ctrl['intensity'] * 100
        de = (r['extent'] - ctrl['extent']) / ctrl['extent'] * 100
        print(f"{exp.upper()}: Intensity={di:+.1f}%, Extent={de:+.1f}%")

# Morning Low-Level (09-12 LT = 13-16 UTC)
analyze_period("MORNING LOW-LEVEL (09-12 LT)", [13, 14, 15, 16], 70, 87)

# Afternoon Upper-Level (15-18 LT = 19-22 UTC)
analyze_period("AFTERNOON UPPER-LEVEL (15-18 LT)", [19, 20, 21, 22], 20, 45)

print("\n" + "="*60)
print("DONE")
print("="*60)
