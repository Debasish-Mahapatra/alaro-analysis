#!/usr/bin/env python3
"""
Decomposition of Updraft Mass Flux into Intensity and Extent Components
========================================================================

This script decomposes the domain-averaged updraft mass flux into two components:

1. EXTENT (Frequency): UD_MESH_FRAC
   - The fractional area of the grid cell covered by updrafts
   - Answers: "Are there fewer/smaller updrafts?"

2. INTENSITY (Amount when present): |UD_OMEGA| where UD_MESH_FRAC > 0
   - The average vertical velocity WITHIN active updrafts (conditional mean)
   - Answers: "Are the updrafts weaker?"

The total mass flux is approximately: Flux ≈ Extent × Intensity

Output:
- panel_decomposition_extent.png: Diurnal cycle of updraft coverage
- panel_decomposition_intensity.png: Diurnal cycle of updraft strength (conditional)
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd

# Configuration
EXPERIMENTS = ['control', '2mom', 'graupel']
INPUT_FILES = {
    'control': 'control_updraft.nc',
    '2mom': '2mom_updraft.nc',
    'graupel': 'graupel_updraft.nc'
}

def load_diurnal_profiles(exp_name):
    """
    Load data and compute the mean diurnal cycle for both components.
    
    Returns:
        intensity: (87 levels, 24 hours) - Conditional mean |UD_OMEGA|
        extent: (87 levels, 24 hours) - Domain mean UD_MESH_FRAC
    """
    print(f"Processing {exp_name}...")
    ds = xr.open_dataset(INPUT_FILES[exp_name])
    
    # Get dimensions
    n_levels = ds.sizes['level']
    n_times = ds.sizes['time']
    
    # Extract variables
    omega = np.abs(ds['UD_OMEGA'].values)   # (time, level, Y, X)
    frac = ds['UD_MESH_FRAC'].values         # (time, level, Y, X)
    
    # Get hour of day in Amazon Local Time (UTC-4)
    time_vals = pd.to_datetime(ds['time'].values)
    hours_local = (time_vals.hour - 4) % 24
    
    # Initialize output arrays
    mean_intensity = np.zeros((n_levels, 24))
    mean_extent = np.zeros((n_levels, 24))
    
    for h in range(24):
        time_mask = (hours_local == h)
        n_samples = np.sum(time_mask)
        
        if n_samples == 0:
            continue
            
        # Select data for this hour
        omega_h = omega[time_mask, :, :, :]  # (n_samples, level, Y, X)
        frac_h = frac[time_mask, :, :, :]
        
        # For each level, compute conditional intensity and domain-mean extent
        for lvl in range(n_levels):
            omega_lvl = omega_h[:, lvl, :, :].flatten()
            frac_lvl = frac_h[:, lvl, :, :].flatten()
            
            # EXTENT: Domain mean of mesh fraction
            mean_extent[lvl, h] = np.nanmean(frac_lvl)
            
            # INTENSITY: Conditional mean (only where updrafts exist)
            active_mask = (frac_lvl > 0)
            if np.sum(active_mask) > 0:
                mean_intensity[lvl, h] = np.nanmean(omega_lvl[active_mask])
            else:
                mean_intensity[lvl, h] = np.nan
    
    ds.close()
    return mean_intensity, mean_extent

def create_panel_plot(data_dict, component, units, filename):
    """
    Create a 3-panel figure: Control (absolute), GRAU-1M-3MT-Control, GRAU-2M-3MT-Control.
    Production quality with proper colorbar positioning.
    """
    # Font sizes
    TITLE_SIZE = 14
    LABEL_SIZE = 13
    TICK_SIZE = 12
    CBAR_LABEL_SIZE = 12
    CBAR_TICK_SIZE = 11
    
    # Use constrained_layout for automatic spacing
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    
    n_levels = data_dict['control'].shape[0]
    hours = np.arange(24)
    levels = np.arange(n_levels)
    
    # Common y-axis label
    YLABEL = 'Model Level (0=Top, 86=Sfc)'
    
    # ---- Panel 1: Control (Absolute) ----
    ax = axes[0]
    ctrl = data_dict['control']
    
    # Use log scale for absolute values
    valid = ctrl[~np.isnan(ctrl) & (ctrl > 0)]
    if len(valid) > 0:
        vmin, vmax = np.percentile(valid, [5, 95])
    else:
        vmin, vmax = 1e-5, 1
    
    pcm0 = ax.pcolormesh(hours, levels, ctrl, 
                         norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                         cmap='hot_r', shading='auto')
    ax.set_title(f'CTRL-1M-3MT ({component})', fontweight='bold', fontsize=TITLE_SIZE)
    ax.set_xlabel('Hour (Amazon UTC-4)', fontsize=LABEL_SIZE)
    ax.set_ylabel(YLABEL, fontsize=LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.invert_yaxis()
    ax.set_xticks([0, 6, 12, 18])
    
    # Colorbar for Panel 1 (below plot)
    cbar0 = fig.colorbar(pcm0, ax=ax, orientation='horizontal', 
                         pad=0.12, shrink=0.9, aspect=25)
    cbar0.set_label(f'{component} [{units}]', fontsize=CBAR_LABEL_SIZE)
    cbar0.ax.tick_params(labelsize=CBAR_TICK_SIZE)
    
    # ---- Panels 2-3: Anomalies ----
    anomalies = {
        '2mom': data_dict['2mom'] - data_dict['control'],
        'graupel': data_dict['graupel'] - data_dict['control']
    }
    
    # Symmetric colorbar based on data range
    all_diffs = np.concatenate([anomalies['2mom'].flatten(), anomalies['graupel'].flatten()])
    all_diffs = all_diffs[~np.isnan(all_diffs)]
    max_abs = np.percentile(np.abs(all_diffs), 99)
    
    for i, (exp, label) in enumerate([('graupel', 'GRAU-1M-3MT'), ('2mom', 'GRAU-2M-3MT')]):
        ax = axes[i + 1]
        diff = anomalies[exp]
        
        pcm = ax.pcolormesh(hours, levels, diff,
                            norm=colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs),
                            cmap='RdBu_r', shading='auto')
        ax.set_title(f'{label} - CTRL-1M-3MT', fontweight='bold', fontsize=TITLE_SIZE)
        ax.set_xlabel('Hour (Amazon UTC-4)', fontsize=LABEL_SIZE)
        ax.set_ylabel(YLABEL, fontsize=LABEL_SIZE)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.invert_yaxis()
        ax.set_xticks([0, 6, 12, 18])
    
    # Single shared colorbar for anomaly panels (below panels 2-3)
    cbar = fig.colorbar(pcm, ax=axes[1:], orientation='horizontal', 
                        pad=0.12, shrink=0.7, aspect=30)
    cbar.set_label(f'{component} Anomaly [{units}]', fontsize=CBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filename}")
    plt.close()

def main():
    print("="*60)
    print("UPDRAFT FLUX DECOMPOSITION ANALYSIS")
    print("="*60)
    
    # Load data for all experiments
    intensity_data = {}
    extent_data = {}
    
    for exp in EXPERIMENTS:
        intensity_data[exp], extent_data[exp] = load_diurnal_profiles(exp)
    
    # Create plots
    print("\nGenerating plots...")
    create_panel_plot(extent_data, 'Updraft Extent', 'Fraction', 
                      'panel_decomposition_extent.png')
    create_panel_plot(intensity_data, 'Updraft Intensity', 'Pa/s', 
                      'panel_decomposition_intensity.png')
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY: Morning Low-Level (09-12 LT, Levels 70-86)")
    print("="*60)
    
    for exp in EXPERIMENTS:
        # Morning = hours 9-12 LT, Low levels = 70-86
        int_morn = intensity_data[exp][70:87, 9:13]
        ext_morn = extent_data[exp][70:87, 9:13]
        print(f"\n{exp.upper()}:")
        print(f"  Mean Intensity: {np.nanmean(int_morn):.3f} Pa/s")
        print(f"  Mean Extent:    {np.nanmean(ext_morn):.5f}")
    
    # Anomalies
    print("\n" + "-"*60)
    print("ANOMALIES (Morning Low-Level):")
    print("-"*60)
    
    ctrl_int = np.nanmean(intensity_data['control'][70:87, 9:13])
    ctrl_ext = np.nanmean(extent_data['control'][70:87, 9:13])
    
    for exp in ['2mom', 'graupel']:
        exp_int = np.nanmean(intensity_data[exp][70:87, 9:13])
        exp_ext = np.nanmean(extent_data[exp][70:87, 9:13])
        
        int_anom = (exp_int - ctrl_int) / ctrl_int * 100
        ext_anom = (exp_ext - ctrl_ext) / ctrl_ext * 100
        
        print(f"\n{exp.upper()} - CONTROL:")
        print(f"  Intensity: {int_anom:+.1f}%")
        print(f"  Extent:    {ext_anom:+.1f}%")
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)

if __name__ == '__main__':
    main()
