#!/usr/bin/env python3
"""
Create panel plots for diurnal cycle evolution of updraft flux.

Figure 1: Side-by-side comparison of absolute flux (3 panels)
Figure 2: Control + Anomalies (GRAU-1M-3MT-Control, GRAU-2M-3MT-Control) (3 panels)

Updraft flux = |UD_OMEGA × UD_MESH_FRAC|
Time: Amazon Local Time (UTC-4)
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
    """Load data and compute mean diurnal cycle for an experiment."""
    input_file = INPUT_FILES[exp_name]
    print(f"Loading {exp_name} from {input_file}...")
    
    ds = xr.open_dataset(input_file)
    
    # Compute updraft flux magnitude
    # Using absolute value as OMEGA is negative for updrafts
    flux = np.abs(ds['UD_OMEGA'] * ds['UD_MESH_FRAC'])
    
    # Get hour of day (Amazon Local Time UTC-4)
    time_vals = pd.to_datetime(ds['time'].values)
    hours_utc = time_vals.hour
    hours = (hours_utc - 4) % 24
    
    n_levels = ds.dims['level']
    
    # Compute mean flux for each hour and level
    mean_flux = np.zeros((n_levels, 24))
    
    for h in range(24):
        mask = (hours == h)
        if np.sum(mask) > 0:
            # We use isel to select time indices first for memory efficiency
            flux_hour = flux.isel(time=mask)
            mean_flux[:, h] = np.nanmean(flux_hour.values, axis=(0, 2, 3))
            
    ds.close()
    return mean_flux

def plot_absolute_panels(data_dict, n_levels):
    """Create 3-panel plot of absolute values."""
    print("Creating absolute comparison plot...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    
    # Determine common color scale (robust)
    # Collect all valid data points
    all_data = []
    for exp in EXPERIMENTS:
        d = data_dict[exp]
        all_data.append(d[d > 1e-20])
    
    all_valid = np.concatenate(all_data)
    vmin = np.percentile(all_valid, 5)
    vmax = np.percentile(all_valid, 95)
    
    hours_grid = np.arange(25) - 0.5
    levels_grid = np.arange(n_levels + 1) - 0.5
    
    for i, exp in enumerate(EXPERIMENTS):
        ax = axes[i]
        mean_flux = data_dict[exp]
        mean_flux_masked = np.ma.masked_where(mean_flux <= 1e-20, mean_flux)
        
        pcm = ax.pcolormesh(
            hours_grid[:-1], levels_grid[:-1], mean_flux_masked,
            norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            cmap='hot_r',
            shading='auto'
        )
        
        ax.set_title(f'{exp.upper()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hour (Amazon UTC-4)', fontsize=12)
        ax.set_xticks(np.arange(0, 24, 6))
        
        if i == 0:
            ax.set_ylabel('Model Level (0=Top, 86=Surface)', fontsize=12)
        
        # Invert Y-axis
        ax.invert_yaxis()
        ax.set_ylim(86.5, -0.5)
        
        # Annotations
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, lw=0.5)
        ax.axhline(y=86, color='gray', linestyle='--', alpha=0.5, lw=0.5)

    # Common colorbar
    plt.tight_layout()
    cbar = fig.colorbar(pcm, ax=axes, orientation='horizontal', fraction=0.08, pad=0.08, shrink=0.8)
    cbar.set_label('Mean Updraft Flux |UD_OMEGA × UD_MESH_FRAC|', fontsize=12)
    
    plt.savefig('panel_comparison_absolute.png', dpi=300, bbox_inches='tight')
    print("Saved panel_comparison_absolute.png")

def plot_anomaly_panels(data_dict, n_levels):
    """Create 3-panel plot: CTRL-1M-3MT, GRAU-1M-3MT-CTRL, GRAU-2M-3MT-CTRL."""
    print("Creating anomaly comparison plot...")
    
    # Font sizes (matching decomposition_analysis.py)
    TITLE_SIZE = 14
    LABEL_SIZE = 13
    TICK_SIZE = 12
    CBAR_LABEL_SIZE = 12
    CBAR_TICK_SIZE = 11
    
    # Common y-axis label
    YLABEL = 'Model Level (0=Top, 86=Sfc)'
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    
    hours_grid = np.arange(25) - 0.5
    levels_grid = np.arange(n_levels + 1) - 0.5
    
    # --- Panel 1: Control (Absolute) ---
    ax = axes[0]
    mean_flux = data_dict['control']
    
    # Scale for absolute
    valid_ctrl = mean_flux[mean_flux > 1e-20]
    vmin_abs = np.percentile(valid_ctrl, 5)
    vmax_abs = np.percentile(valid_ctrl, 95)
    
    pcm_abs = ax.pcolormesh(
        hours_grid[:-1], levels_grid[:-1], np.ma.masked_where(mean_flux <= 1e-20, mean_flux),
        norm=colors.LogNorm(vmin=vmin_abs, vmax=vmax_abs),
        cmap='hot_r',
        shading='auto'
    )
    ax.set_title('CTRL-1M-3MT (Absolute)', fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xlabel('Hour (Amazon UTC-4)', fontsize=LABEL_SIZE)
    ax.set_ylabel(YLABEL, fontsize=LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.set_xticks(np.arange(0, 24, 6))
    ax.invert_yaxis()
    ax.set_ylim(86.5, -0.5)
    
    # --- Anomalies ---
    anomalies = {
        '2mom': data_dict['2mom'] - data_dict['control'],
        'graupel': data_dict['graupel'] - data_dict['control']
    }
    
    # Determine symmetric scale for anomalies
    all_diffs = np.concatenate([anomalies['2mom'].flatten(), anomalies['graupel'].flatten()])
    max_diff_robust = np.percentile(np.abs(all_diffs), 98)
    norm_diff = colors.TwoSlopeNorm(vmin=-max_diff_robust, vcenter=0, vmax=max_diff_robust)
    
    # Panels 2 & 3: Changed order to graupel first, then 2mom
    exp_labels = [('graupel', 'GRAU-1M-3MT'), ('2mom', 'GRAU-2M-3MT')]
    for i, (exp, label) in enumerate(exp_labels):
        ax = axes[i+1]
        diff = anomalies[exp]
        
        pcm_diff = ax.pcolormesh(
            hours_grid[:-1], levels_grid[:-1], diff,
            norm=norm_diff,
            cmap='RdBu_r',
            shading='auto'
        )
        
        ax.set_title(f'{label} - CTRL-1M-3MT', fontsize=TITLE_SIZE, fontweight='bold')
        ax.set_xlabel('Hour (Amazon UTC-4)', fontsize=LABEL_SIZE)
        ax.set_ylabel(YLABEL, fontsize=LABEL_SIZE)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.set_xticks(np.arange(0, 24, 6))
        ax.invert_yaxis()
        ax.set_ylim(86.5, -0.5)

    # Colorbar for Absolute (under Panel 1)
    cbar1 = fig.colorbar(pcm_abs, ax=axes[0], orientation='horizontal', pad=0.12, shrink=0.9)
    cbar1.set_label('Mean Flux (Absolute)', fontsize=CBAR_LABEL_SIZE)
    cbar1.ax.tick_params(labelsize=CBAR_TICK_SIZE)
    
    # Colorbar for Anomalies (under Panels 2 & 3)
    cbar2 = fig.colorbar(pcm_diff, ax=axes[1:], orientation='horizontal', pad=0.12, shrink=0.6)
    cbar2.set_label('Flux Anomaly (Experiment - Control)', fontsize=CBAR_LABEL_SIZE)
    cbar2.ax.tick_params(labelsize=CBAR_TICK_SIZE)
    
    plt.savefig('panel_comparison_anomaly.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved panel_comparison_anomaly.png")

def main():
    print("Starting panel plot generation...")
    
    # Load all data
    data_dict = {}
    n_levels = 87
    
    for exp in EXPERIMENTS:
        data_dict[exp] = load_and_process_data(exp)
        n_levels = data_dict[exp].shape[0]
        
    # Plot 1: Absolute Comparison
    plot_absolute_panels(data_dict, n_levels)
    
    # Plot 2: Anomalies
    plot_anomaly_panels(data_dict, n_levels)
    
    print("\nCOMPLETE!")

if __name__ == '__main__':
    main()
