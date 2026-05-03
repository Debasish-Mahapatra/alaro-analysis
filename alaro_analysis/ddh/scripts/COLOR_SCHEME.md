# Budget Plot Color Scheme

## Overview
- **One color per physical process** across all plots
- **Resolved processes**: Dashed line
- **Parametrized/Convective processes**: Dash-dot line
- **0 °C isotherm**: Dotted neutral grey line
- This ensures consistency and easy visual distinction

## Process Color Map

| Process | Color | Hex Code |
|---------|-------|----------|
| Dynamics | Blue | #1B6EC2 |
| Micro | Red | #D93425 |
| Condensation | Green | #228B22 |
| Evaporation | Purple | #9B59B6 |
| Autoconversion | Orange | #E8891D |
| Precipitation | Bright Red | #FF6B6B |
| Turbulence (diff) | Teal | #00A6A6 |
| Turbulence (conv) | Magenta | #E84393 |
| Radiation (solar) | Gold | #FFD700 |
| Radiation (thermal) | Emerald | #17B890 |
| Shear | Dark Purple | #6C3483 |
| Buoyancy | Amber | #F39C12 |
| Dissipation | Dark Red | #C70039 |
| Advection | Navy Blue | #0066CC |
| Diffusion | Dark Orange | #FF8C00 |
| GWD drag | Light Sea Green | #20B2AA |
| Negativity correction | Saddle Brown | #8B4513 |

## Line Style Convention

- **Resolved** (physical/grid-scale): dashed line, α=0.95
- **Parametrized/Conv** (sub-grid-scale): dash-dot line, α=0.85
- **0 °C isotherm**: dotted neutral grey line, α=0.95
- **Partition figures**: convection uses red, resolved uses blue, and total uses black
- **Dynamics/Single variant**: `—` solid line, α=0.92
- **Tendency**: `—` solid black, α=1.0, linewidth=2.4
- **Residual**: `· · ·` dotted grey, α=0.85
- **Sum of tendencies**: `- · -` dash-dot grey, α=0.85

## Examples

### QI, QL, QR, QS Budgets
- Condensation (resolved) = Green dashed
- Condensation (conv) = Green dash-dot
- Autoconversion (resolved) = Orange dashed
- Autoconversion (conv) = Orange dash-dot
- Precipitation (resolved) = Bright Red dashed
- Precipitation (conv) = Bright Red dash-dot
- Evaporation (resolved) = Purple dashed
- Evaporation (conv) = Purple dash-dot

### CT Budget
- Dynamics = Blue solid
- Micro (resolved) = Red dashed
- Micro (convective) = Red solid (no dashed variant in this context)

### QV Budget
- Dynamics = Blue solid
- Condensation (resolved) = Green dashed
- Condensation (conv) = Green dash-dot
- Evaporation (resolved) = Purple dashed
- Evaporation (conv) = Purple dash-dot

## Implementation Details

The color scheme is defined in `alaro_analysis/ddh/plot_style.py`:
- `PROCESS_COLORS` / `PROCESS_COLOURS`: Maps process name → hex color code
- `get_process_name()`: Extracts process name from full label
- `get_line_style()`: Returns (color, linewidth, linestyle, alpha, zorder)
- `partition_line_style()`: Styles total/convection/resolved partition curves
- `pathway_from_block()` / `pathway_line_attributes()`: Styles raw DDH block names like `cond-cv` and `cond-rs`
  - Checks if label contains "resolved" → dashed line
  - Checks if label contains "conv" → dash-dot line
  - Otherwise → solid line

`alaro_analysis/ddh/io.py` derives `BLOCK_COLORS` from this same palette so
case-study and budget plots do not silently drift away from the chosen colors.
