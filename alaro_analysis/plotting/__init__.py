"""Plotting entrypoints. Importing scales does not initialize panel backends."""

from importlib import import_module

_EXPORTS = {
    "plot_surface_diurnal_cycle": ".panels",
    "plot_three_panel_diurnal": ".panels",
    "infer_abs_limits": ".scales",
    "infer_anom_scale": ".scales",
    "robust_anomaly_scale": ".scales",
    "robust_log_limits": ".scales",
    "resolve_workers": ".style",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
