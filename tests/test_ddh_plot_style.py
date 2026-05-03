from alaro_analysis.ddh.plot_style import (
    CONVECTION_LINESTYLE,
    CONVECTION_COLOR,
    FREEZING_COLOR,
    FREEZING_LINESTYLE,
    PROCESS_COLORS,
    RESOLVED_COLOR,
    RESOLVED_LINESTYLE,
    get_line_style,
    get_process_name,
    pathway_from_block,
    pathway_line_attributes,
    partition_line_style,
)
from alaro_analysis.ddh.io import BLOCK_COLORS
from alaro_analysis.ddh.scripts import plot_time_average


def test_process_color_convention_is_shared_with_time_average_script():
    assert PROCESS_COLORS["Condensation"] == "#228B22"
    assert PROCESS_COLORS["Evaporation"] == "#9B59B6"
    assert plot_time_average.PROCESS_COLOURS["Condensation"] == "#228B22"
    assert BLOCK_COLORS["cond-cv"] == "#228B22"
    assert BLOCK_COLORS["evap-rs"] == "#9B59B6"


def test_line_style_convention_separates_process_from_pathway():
    assert get_process_name("Condensation (resolved)") == "Condensation"

    color, _lw, linestyle, alpha, _zorder = get_line_style("Condensation (resolved)")
    assert color == "#228B22"
    assert linestyle == RESOLVED_LINESTYLE
    assert alpha == 0.95

    color, _lw, linestyle, alpha, _zorder = get_line_style("Condensation (conv)")
    assert color == "#228B22"
    assert linestyle == CONVECTION_LINESTYLE
    assert alpha == 0.85


def test_partition_line_style_uses_same_process_colors():
    color, _lw, linestyle, alpha, zorder = partition_line_style(
        "Evaporation",
        "convection",
    )
    assert color == CONVECTION_COLOR
    assert linestyle == CONVECTION_LINESTYLE
    assert alpha == 0.85
    assert zorder > 2

    color, _lw, linestyle, alpha, zorder = partition_line_style(
        "Evaporation",
        "resolved",
    )
    assert color == RESOLVED_COLOR
    assert linestyle == RESOLVED_LINESTYLE
    assert alpha == 0.95
    assert zorder > 2


def test_block_pathway_helpers_support_raw_ddh_names():
    assert pathway_from_block("cond-cv") == "convection"
    assert pathway_from_block("condrs") == "resolved"

    linestyle, alpha = pathway_line_attributes(pathway_from_block("evap-cv"))
    assert linestyle == CONVECTION_LINESTYLE
    assert alpha == 0.85


def test_freezing_line_uses_neutral_dotted_style():
    assert FREEZING_COLOR == "#555555"
    assert FREEZING_LINESTYLE == ":"
