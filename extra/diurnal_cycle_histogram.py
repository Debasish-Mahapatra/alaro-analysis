#!/usr/bin/env python3
"""
Create diurnal cycle evolution plot of updraft flux for multiple experiments.
Updraft flux = |UD_OMEGA × UD_MESH_FRAC|

2D Hovmöller-style plot:
- X-axis: Hour of day (Amazon UTC-4)
- Y-axis: Vertical level (0=model top, 86=near surface)
- Color: Mean updraft flux magnitude
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
OUTPUT_FILES = {
    'control': 'control_diurnal_cycle_vertical.png',
    '2mom': '2mom_diurnal_cycle_vertical.png',
    'graupel': 'graupel_diurnal_cycle_vertical.png'
}

def process_experiment(exp_name):
    input_file = INPUT_FILES[exp_name]
    output_file = OUTPUT_FILES[exp_name]
    
    print("\n" + "="*70)
    print(f"Processing {exp_name.upper()} Experiment")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print("="*70)
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return
        
    # Load data
    print("Loading data...")
    ds = xr.open_dataset(input_file)
    
    # Compute updraft flux magnitude
    print("Computing updraft flux |UD_OMEGA × UD_MESH_FRAC|...")
    flux = np.abs(ds['UD_OMEGA'] * ds['UD_MESH_FRAC'])
    
    # Get hour of day (Amazon Local Time UTC-4)
    print("Extracting time (UTC-4)...")
    time_vals = pd.to_datetime(ds['time'].values)
    hours_utc = time_vals.hour
    hours = (hours_utc - 4) % 24
    
    # Get dimensions
    n_levels = ds.dims['level']
    n_timesteps = len(hours)
    print(f"Levels: {n_levels}")
    print(f"Timesteps: {n_timesteps}")
    
    # Compute mean flux for each hour and level
    print("Computing mean flux by hour and level...")
    mean_flux = np.zeros((n_levels, 24))
    
    for h in range(24):
        mask = (hours == h)
        if np.sum(mask) > 0:
            # Mean over time, Y, X for each level
            # We can select time indices first to reduce memory usage
            flux_hour = flux.isel(time=mask)
            mean_flux[:, h] = np.nanmean(flux_hour.values, axis=(0, 2, 3))
    
    ds.close()
    
    # Create plot
    create_plot(mean_flux, n_levels, exp_name, output_file)

def create_plot(mean_flux, n_levels, exp_name, output_file):
    print("Creating plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create meshgrid for pcolormesh
    hours_grid = np.arange(25) - 0.5  # Hour edges
    levels_grid = np.arange(n_levels + 1) - 0.5  # Level edges
    
    # Mask zeros/negative values for log scale
    mean_flux_masked = np.ma.masked_where(mean_flux <= 1e-20, mean_flux)
    
    # Determine color scale limits (robust)
    valid_data = mean_flux[mean_flux > 1e-20]
    if len(valid_data) > 0:
        vmin = np.percentile(valid_data, 5)
        vmax = np.percentile(valid_data, 95)
    else:
        vmin, vmax = 1e-10, 1e-5
        
    # Plot
    pcm = ax.pcolormesh(
        hours_grid[:-1], levels_grid[:-1], mean_flux_masked,
        norm=colors.LogNorm(vmin=vmin, vmax=vmax),
        cmap='hot_r',
        shading='auto'
    )
    
    # Colorbar
    cbar = plt.colorbar(pcm, ax=ax, label='Mean Updraft Flux |UD_OMEGA × UD_MESH_FRAC|', 
                        shrink=0.9, pad=0.02)
    
    # Labels and formatting
    ax.set_xlabel('Hour of Day (Amazon Local Time, UTC-4)', fontsize=14)
    ax.set_ylabel('Model Level (0=Top, 86=Surface)', fontsize=14)
    ax.set_title(f'{exp_name.upper()} Experiment: Diurnal Cycle of Updraft Flux\nManaus Region, 2014-2015', 
                 fontsize=16, fontweight='bold')
    
    # X-axis ticks
    ax.set_xticks(np.arange(0, 24, 2))
    ax.set_xlim(-0.5, 23.5)
    
    # Y-axis: invert so surface is at bottom
    ax.invert_yaxis()
    ax.set_ylim(86.5, -0.5)
    
    # Annotations
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.axhline(y=86, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.text(24.5, 0, 'Model Top', fontsize=10, va='center', ha='left')
    ax.text(24.5, 86, 'Surface', fontsize=10, va='center', ha='left')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=400, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    # Statistics
    print("\nStatistics:")
    print(f"  Min flux (displayed): {vmin:.2e}")
    print(f"  Max flux (displayed): {vmax:.2e}")
    print(f"  Peak hour (local): {np.argmax(np.nanmean(mean_flux, axis=0))}")
    print(f"  Peak level: {np.argmax(np.nanmean(mean_flux, axis=1))}")

def main():
    print("Starting batch processing...")
    for exp in EXPERIMENTS:
        process_experiment(exp)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)

if __name__ == '__main__':
    main()
