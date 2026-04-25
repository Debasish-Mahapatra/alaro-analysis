"""Data discovery, cache, and IO helpers."""

_EXPORTS = {
    "build_cache_file": ".cache",
    "build_diurnal_cache_file": ".cache",
    "build_height_cache_file": ".cache",
    "load_cache": ".cache",
    "save_cache": ".cache",
    "nanmean_with_count": ".dataset_io",
    "read_time_level_yx": ".dataset_io",
    "read_vertical_profile": ".dataset_io",
    "resolve_data_var_name": ".dataset_io",
    "select_data_var_name": ".dataset_io",
    "to_time_level_yx": ".dataset_io",
    "collect_file_records": ".discovery",
    "discover_variable_maps": ".discovery",
    "discover_variables": ".discovery",
    "resolve_var_name": ".discovery",
}

__all__ = [
    "build_cache_file",
    "build_diurnal_cache_file",
    "build_height_cache_file",
    "collect_file_records",
    "discover_variable_maps",
    "discover_variables",
    "resolve_var_name",
    "load_cache",
    "nanmean_with_count",
    "read_time_level_yx",
    "read_vertical_profile",
    "resolve_data_var_name",
    "save_cache",
    "select_data_var_name",
    "to_time_level_yx",
]


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'alaro_analysis.data' has no attribute {name!r}")

    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
