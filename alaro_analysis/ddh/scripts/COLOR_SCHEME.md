# Budget Plot Color Scheme

## Overview
- **One color per physical process** across all plots
- **Resolved processes**: Solid line (`-`)
- **Parametrized/Convective processes**: Dashed line (`- -`)
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

- **Resolved** (physical/grid-scale): `—` solid line, α=0.95
- **Parametrized/Conv** (sub-grid-scale): `- -` dashed line, α=0.85
- **Dynamics/Single variant**: `—` solid line, α=0.92
- **Tendency**: `—` solid black, α=1.0, linewidth=2.4
- **Residual**: `· · ·` dotted grey, α=0.85
- **Sum of tendencies**: `- · -` dash-dot grey, α=0.85

## Examples

### QI, QL, QR, QS Budgets
- Condensation (resolved) = Green solid
- Condensation (conv) = Green dashed
- Autoconversion (resolved) = Orange solid
- Autoconversion (conv) = Orange dashed
- Precipitation (resolved) = Bright Red solid
- Precipitation (conv) = Bright Red dashed
- Evaporation (resolved) = Purple solid
- Evaporation (conv) = Purple dashed

### CT Budget
- Dynamics = Blue solid
- Micro (resolved) = Red solid
- Micro (convective) = Red solid (no dashed variant in this context)

### QV Budget
- Dynamics = Blue solid
- Condensation (resolved) = Green solid
- Condensation (conv) = Green dashed
- Evaporation (resolved) = Purple solid
- Evaporation (conv) = Purple dashed

## Implementation Details

The color scheme is defined in `plot_time_average.py`:
- `PROCESS_COLOURS`: Maps process name → hex color code
- `get_process_name()`: Extracts process name from full label
- `get_line_style()`: Returns (color, linewidth, linestyle, alpha, zorder)
  - Checks if label contains "resolved" → solid line
  - Checks if label contains "conv" → dashed line
  - Otherwise → solid line

