#!/usr/bin/env python3
"""Detailed verification of all values with full transparency."""
import xarray as xr
import numpy as np
import pandas as pd

files = {
    'control': 'control_updraft.nc',
    '2mom': '2mom_updraft.nc',
    'graupel': 'graupel_updraft.nc'
}

def analyze_period(name, hours_local, level_start, level_end):
    """
    Analyze a specific period using LOCAL TIME hours.
    hours_local: list of local time hours (Amazon UTC-4)
    """
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"Local Hours: {hours_local}, Levels: {level_start}-{level_end}")
    print('='*70)
    
    results = {}
    for exp, fname in files.items():
        ds = xr.open_dataset(fname)
        omega = np.abs(ds['UD_OMEGA'].values)
        frac = ds['UD_MESH_FRAC'].values
        
        # Convert UTC to Local Time (UTC-4)
        times = pd.to_datetime(ds['time'].values)
        hours_local_all = (times.hour - 4) % 24
        
        # Select matching hours
        mask_time = np.isin(hours_local_all, hours_local)
        n_samples = np.sum(mask_time)
        
        omega_sub = omega[mask_time, level_start:level_end, :, :]
        frac_sub = frac[mask_time, level_start:level_end, :, :]
        
        # Extent: domain mean
        ext_mean = np.nanmean(frac_sub)
        
        # Intensity: conditional mean where frac > 0
        mask_active = (frac_sub > 0)
        n_active = np.sum(mask_active)
        if n_active > 0:
            int_mean = np.nanmean(omega_sub[mask_active])
        else:
            int_mean = np.nan
        
        results[exp] = {'intensity': int_mean, 'extent': ext_mean, 
                        'n_samples': n_samples, 'n_active': n_active}
        ds.close()
    
    print("\nAbsolute Values:")
    for exp in ['control', '2mom', 'graupel']:
        r = results[exp]
        print(f"  {exp.upper():10s}: Int={r['intensity']:.4f} Pa/s, Ext={r['extent']:.6f} "
              f"(samples={r['n_samples']}, active_pts={r['n_active']})")
    
    print("\nAnomalies (% change from Control):")
    ctrl = results['control']
    for exp in ['2mom', 'graupel']:
        r = results[exp]
        di = (r['intensity'] - ctrl['intensity']) / ctrl['intensity'] * 100
        de = (r['extent'] - ctrl['extent']) / ctrl['extent'] * 100
        print(f"  {exp.upper():10s}: Intensity = {di:+.1f}%,  Extent = {de:+.1f}%")

# ============================================================
# MORNING LOW-LEVEL
# ============================================================
# Local hours 9, 10, 11, 12 (09:00-12:59 LT)
# Levels 70-86 (indices 70:87)
analyze_period("MORNING LOW-LEVEL", [9, 10, 11, 12], 70, 87)

# ============================================================
# AFTERNOON UPPER-LEVEL
# ============================================================
# Local hours 15, 16, 17, 18 (15:00-18:59 LT)
# Levels 20-44 (indices 20:45)
analyze_period("AFTERNOON UPPER-LEVEL", [15, 16, 17, 18], 20, 45)

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
