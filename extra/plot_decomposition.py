#!/usr/bin/env python3
"""
Decompose Updraft Flux into Intensity and Extent components.

Updraft Flux = Intensity × Extent
Intensity = |UD_OMEGA| (Pa/s)
Extent = UD_MESH_FRAC (fraction 0-1)

Generates 3-panel plots for each component:
1. Intensity (Control, 2mom-Control, Graupel-Control)
2. Extent (Control, 2mom-Control, Graupel-Control)
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import os

# Configuration
EXPERIMENTS = ['control', '2mom', 'graupel']
INPUT_FILES = {
    'control': 'control_updraft.nc',
    '2mom': '2mom_updraft.nc',
    'graupel': 'graupel_updraft.nc'
}

def load_and_process_data(exp_name):
    """Load data and compute mean diurnal cycle for components."""
    input_file = INPUT_FILES[exp_name]
    print(f"Loading {exp_name} from {input_file}...")
    
    ds = xr.open_dataset(input_file)
    
    # Components
    # Intensity: |UD_OMEGA|
    intensity = np.abs(ds['UD_OMEGA'])
    # Extent: UD_MESH_FRAC
    extent = ds['UD_MESH_FRAC']
    
    # Get hour of day (Amazon Local Time UTC-4)
    time_vals = pd.to_datetime(ds['time'].values)
    hours_utc = time_vals.hour
    hours = (hours_utc - 4) % 24
    
    n_levels = ds.dims['level']
    
    # Compute means
    mean_intensity = np.zeros((n_levels, 24))
    mean_extent = np.zeros((n_levels, 24))
    
    for h in range(24):
        mask = (hours == h)
        if np.sum(mask) > 0:
            # Select time slice
            int_hour = intensity.isel(time=mask)
            ext_hour = extent.isel(time=mask)
            
            # Mean over time, Y, X
            # IMPORTANT: For "Conditional Intensity", we must only average where there are active updrafts
            # UD_OMEGA is 0 where UD_MESH_FRAC is 0 (verified)
            # So we mask intensity where extent == 0
            
            mask_active = (ext_hour > 0)
            
            # Correct calculation: Loop over levels to get per-level conditional mean
            for lvl in range(n_levels):
                # Select data for this level
                int_lvl = int_hour.isel(level=lvl).values
                ext_lvl = ext_hour.isel(level=lvl).values
                
                # Mask where this specific level has updrafts
                mask_lvl = (ext_lvl > 0)
                
                # Conditional mean for this level
                active_vals = int_lvl[mask_lvl]
                
                if len(active_vals) > 0:
                    mean_intensity[lvl, h] = np.mean(active_vals)
                else:
                    mean_intensity[lvl, h] = np.nan
            
            # Extent is correctly averaged over the whole domain (grid fraction)
            # axis=(0,2,3) reduces (time, Y, X), leaving (level,)
            mean_extent[:, h] = np.nanmean(ext_hour.values, axis=(0, 2, 3))
            
    ds.close()
    return mean_intensity, mean_extent

def plot_panels(data_dict, component_name, unit, title_prefix, filename, log_scale=True):
    """Create 3-panel plot (Control, Anomaly1, Anomaly2)."""
    print(f"Creating {component_name} plot...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    
    n_levels = data_dict['control'].shape[0]
    hours_grid = np.arange(25) - 0.5
    levels_grid = np.arange(n_levels + 1) - 0.5
    
    # --- Panel 1: Control (Absolute) ---
    ax = axes[0]
    data_ctrl = data_dict['control']
    
    # Scale
    valid_data = data_ctrl[data_ctrl > 1e-20]
    if len(valid_data) > 0:
        vmin = np.percentile(valid_data, 5)
        vmax = np.percentile(valid_data, 95)
    else:
        vmin, vmax = 1e-5, 1.0
    
    norm = colors.LogNorm(vmin=vmin, vmax=vmax) if log_scale else colors.Normalize(vmin=0, vmax=vmax)
    
    pcm_abs = ax.pcolormesh(
        hours_grid[:-1], levels_grid[:-1], 
        np.ma.masked_where(data_ctrl <= 1e-20, data_ctrl),
        norm=norm,
        cmap='hot_r',
        shading='auto'
    )
    ax.set_title(f'CONTROL ({component_name})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Hour (Amazon UTC-4)', fontsize=12)
    ax.set_ylabel('Model Level (0=Top, 86=Surface)', fontsize=12)
    ax.set_xticks(np.arange(0, 24, 6))
    ax.invert_yaxis()
    ax.set_ylim(86.5, -0.5)
    
    # --- Anomalies ---
    anomalies = {
        '2mom': data_dict['2mom'] - data_dict['control'],
        'graupel': data_dict['graupel'] - data_dict['control']
    }
    
    # Scale for anomalies
    all_diffs = np.concatenate([anomalies['2mom'].flatten(), anomalies['graupel'].flatten()])
    # Robust max (98th percentile to avoid outliers) or absolute max if preferred
    # User requested "-max to +max", often implies absolute max or a robust max cover.
    # Let's use robust max to avoid single-pixel outliers skewing the plot
    max_diff = np.percentile(np.abs(all_diffs), 99) 
    
    # If user wants absolute max, uncomment below:
    # max_diff = np.max(np.abs(all_diffs))
    
    norm_diff = colors.TwoSlopeNorm(vmin=-max_diff, vcenter=0, vmax=max_diff)
    
    for i, exp in enumerate(['2mom', 'graupel']):
        ax = axes[i+1]
        diff = anomalies[exp]
        
        pcm_diff = ax.pcolormesh(
            hours_grid[:-1], levels_grid[:-1], diff,
            norm=norm_diff,
            cmap='RdBu_r', 
            shading='auto'
        )
        
        ax.set_title(f'{exp.upper()} - CONTROL', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hour (Amazon UTC-4)', fontsize=12)
        ax.set_xticks(np.arange(0, 24, 6))
        ax.invert_yaxis()
        ax.set_ylim(86.5, -0.5)
        
        # Annotations
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, lw=0.5)
        ax.axhline(y=86, color='gray', linestyle='--', alpha=0.5, lw=0.5)

    plt.tight_layout()
    
    # Colorbars
    cbar1 = fig.colorbar(pcm_abs, ax=axes[0], orientation='horizontal', pad=0.15, shrink=0.9)
    cbar1.set_label(f'{component_name} [{unit}]', fontsize=10)
    
    cbar2 = fig.colorbar(pcm_diff, ax=axes[1:], orientation='horizontal', pad=0.15, shrink=0.6)
    cbar2.set_label(f'{component_name} Anomaly [{unit}]', fontsize=10)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")

def main():
    print("Starting decomposition analysis...")
    
    intensity_data = {}
    extent_data = {}
    
    for exp in EXPERIMENTS:
        i, e = load_and_process_data(exp)
        intensity_data[exp] = i
        extent_data[exp] = e
    
    # Plot Intensity (|UD_OMEGA|)
    plot_panels(intensity_data, 'Updraft Intensity', 'Pa/s', 'Intensity', 
               'panel_decomposition_intensity.png', log_scale=True)
    
    # Plot Extent (UD_MESH_FRAC)
    plot_panels(extent_data, 'Updraft Extent', 'Fraction', 'Extent', 
               'panel_decomposition_extent.png', log_scale=True)
    
    print("\nCOMPLETE!")

if __name__ == '__main__':
    main()
