from __future__ import annotations

from .models import SpatialWindow


def parse_slice_arg(spec: str | None, axis_name: str) -> tuple[int | None, int | None]:
    if spec is None:
        return None, None
    token = spec.strip()
    if ":" not in token:
        raise ValueError(
            f"Invalid --{axis_name}-slice '{spec}'. Expected 'start:end' (Python slice)."
        )
    left, right = token.split(":", 1)
    start = int(left) if left.strip() else None
    end = int(right) if right.strip() else None
    if start is not None and end is not None and end <= start:
        raise ValueError(
            f"Invalid --{axis_name}-slice '{spec}': end must be greater than start."
        )
    return start, end


def build_spatial_window(y_slice: str | None, x_slice: str | None) -> SpatialWindow:
    y_start, y_end = parse_slice_arg(y_slice, "y")
    x_start, x_end = parse_slice_arg(x_slice, "x")
    return SpatialWindow(y_start=y_start, y_end=y_end, x_start=x_start, x_end=x_end)


def spatial_window_tag(spatial_window: SpatialWindow) -> str:
    if (
        spatial_window.y_start is None
        and spatial_window.y_end is None
        and spatial_window.x_start is None
        and spatial_window.x_end is None
    ):
        return "full-domain"

    def bound(value: int | None) -> str:
        return "none" if value is None else str(int(value))

    return (
        f"y{bound(spatial_window.y_start)}-{bound(spatial_window.y_end)}_"
        f"x{bound(spatial_window.x_start)}-{bound(spatial_window.x_end)}"
    )
