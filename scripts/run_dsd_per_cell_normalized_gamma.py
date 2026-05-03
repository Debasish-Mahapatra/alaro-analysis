"""Plot per-cell DSD samples with all panels in normalized-gamma space.

The expensive model per-cell sampling is reused from
``disdrometer_dsd_percell_samples_all_leads.npz``.  The model samples already
come from ALARO DSD assumptions: C1M/G1M are Abel-Boutle exponential
distributions (normalized-gamma ``mu=0``), while G2M is gamma with ``mu=1``.

The observation samples in that NPZ are empirical Path A diagnostics.  This
script fits those observations into the normalized-gamma family by estimating
``mu`` from the empirical mass-spectrum width, then recomputing D0, sigma_m,
Nt, and log10(Nw) from Dm, LWC, and mu.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from alaro_analysis.common.constants import EXPERIMENTS, EXPERIMENT_LABELS
from alaro_analysis.common.dsd import (
    normalized_gamma_diagnostics_from_lwc_dm_mu,
    normalized_gamma_diagnostics_from_lwc_nt_mu,
    normalized_gamma_from_empirical_samples,
)
from alaro_analysis.workflows.disdrometer_dsd import (
    DEFAULT_FIGURE_DPI,
    FIGURE_DIR,
    OBS_PARAMETERS,
    PANEL_GRID_POSITIONS,
    PANEL_ORDER,
    PROCESSED_DIR,
    _compute_obs_contour,
    plot_2x2_with_marginals,
)


def _load_prefixed(npz: np.lib.npyio.NpzFile, prefix: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for key in OBS_PARAMETERS:
        npz_key = f"{prefix}__{key}"
        if npz_key not in npz.files:
            missing.append(npz_key)
        else:
            out[key] = np.asarray(npz[npz_key])
    if missing:
        raise KeyError(f"Missing keys in samples NPZ: {', '.join(missing)}")
    return out


def _finite_count(samples: dict[str, np.ndarray]) -> int:
    keep = np.isfinite(samples["dm_mm"]) & np.isfinite(samples["log_nw"])
    return int(np.count_nonzero(keep))


def _normalized_gamma_fixed_mu(
    samples: dict[str, np.ndarray],
    mu: float,
) -> dict[str, np.ndarray]:
    diag = normalized_gamma_diagnostics_from_lwc_dm_mu(
        samples["lwc_g_m3"],
        samples["dm_mm"],
        mu,
    )
    keep = (
        np.isfinite(diag["dm_mm"]) & (diag["dm_mm"] > 0.0)
        & np.isfinite(diag["d0_mm"]) & (diag["d0_mm"] > 0.0)
        & np.isfinite(diag["sigma_m_mm"]) & (diag["sigma_m_mm"] > 0.0)
        & np.isfinite(diag["log_nw"])
        & np.isfinite(diag["lwc_g_m3"]) & (diag["lwc_g_m3"] > 0.0)
        & np.isfinite(diag["nt_m3"]) & (diag["nt_m3"] > 0.0)
    )
    keys = (*OBS_PARAMETERS, "mu")
    return {key: diag[key][keep].astype(np.float32) for key in keys}


def _normalized_gamma_fixed_mu_from_lwc_nt(
    samples: dict[str, np.ndarray],
    mu: float,
) -> dict[str, np.ndarray]:
    diag = normalized_gamma_diagnostics_from_lwc_nt_mu(
        samples["lwc_g_m3"],
        samples["nt_m3"],
        mu,
    )
    keep = (
        np.isfinite(diag["dm_mm"]) & (diag["dm_mm"] > 0.0)
        & np.isfinite(diag["d0_mm"]) & (diag["d0_mm"] > 0.0)
        & np.isfinite(diag["sigma_m_mm"]) & (diag["sigma_m_mm"] > 0.0)
        & np.isfinite(diag["log_nw"])
        & np.isfinite(diag["lwc_g_m3"]) & (diag["lwc_g_m3"] > 0.0)
        & np.isfinite(diag["nt_m3"]) & (diag["nt_m3"] > 0.0)
    )
    keys = (*OBS_PARAMETERS, "mu")
    return {key: diag[key][keep].astype(np.float32) for key in keys}


def _with_constant_mu(samples: dict[str, np.ndarray], mu: float) -> dict[str, np.ndarray]:
    out = dict(samples)
    out["mu"] = np.full(samples["dm_mm"].shape, float(mu), dtype=np.float32)
    return out


def _field_values(samples: dict[str, np.ndarray], field: str) -> np.ndarray:
    if field == "d0_over_dm":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(samples["d0_mm"], dtype=float) / np.asarray(samples["dm_mm"], dtype=float)
    if field == "sigma_over_dm":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(samples["sigma_m_mm"], dtype=float) / np.asarray(samples["dm_mm"], dtype=float)
    return np.asarray(samples[field], dtype=float)


def _plot_2x2_generic_with_marginals(
    out_path: Path,
    samples: dict[str, dict[str, np.ndarray]],
    x_field: str,
    y_field: str,
    x_label: str,
    y_label: str,
    *,
    title: str,
    bins: int,
    min_x: float | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_los: list[float] = []
    x_his: list[float] = []
    y_los: list[float] = []
    y_his: list[float] = []
    for name in PANEL_ORDER:
        sx = _field_values(samples[name], x_field)
        sy = _field_values(samples[name], y_field)
        ok = np.isfinite(sx) & np.isfinite(sy)
        if not ok.any():
            continue
        x_los.append(float(np.nanpercentile(sx[ok], 0.5)))
        x_his.append(float(np.nanpercentile(sx[ok], 99.5)))
        y_los.append(float(np.nanpercentile(sy[ok], 0.5)))
        y_his.append(float(np.nanpercentile(sy[ok], 99.5)))
    if not x_los or not y_los:
        raise RuntimeError(f"No samples available to plot {x_field} vs {y_field}")

    x_lo = min(x_los)
    if min_x is not None:
        x_lo = max(min_x, x_lo)
    x_hi = max(max(x_his), x_lo + 0.5)
    y_lo = min(y_los)
    y_hi = max(max(y_his), y_lo + 0.5)
    x_pad = 0.04 * (x_hi - x_lo)
    y_pad = 0.04 * (y_hi - y_lo)
    x_lo -= x_pad
    x_hi += x_pad
    y_lo -= y_pad
    y_hi += y_pad
    x_edges = np.linspace(x_lo, x_hi, bins + 1)
    y_edges = np.linspace(y_lo, y_hi, bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    obs_x = _field_values(samples["obs"], x_field)
    obs_y = _field_values(samples["obs"], y_field)
    contour = _compute_obs_contour(obs_x[np.isfinite(obs_x) & np.isfinite(obs_y)], obs_y[np.isfinite(obs_x) & np.isfinite(obs_y)], x_edges, y_edges)

    color_for_name = {
        "obs": "#111111",
        "control": "#d62728",
        "graupel": "#1f77b4",
        "2mom": "#2ca02c",
    }
    label_for_name = {
        "obs": "Obs",
        "control": EXPERIMENT_LABELS.get("control", "C1M"),
        "graupel": EXPERIMENT_LABELS.get("graupel", "G1M"),
        "2mom": EXPERIMENT_LABELS.get("2mom", "G2M"),
    }

    x_pdf: dict[str, np.ndarray] = {}
    y_pdf: dict[str, np.ndarray] = {}
    for name in PANEL_ORDER:
        sx = _field_values(samples[name], x_field)
        sy = _field_values(samples[name], y_field)
        ok = np.isfinite(sx) & np.isfinite(sy)
        x_pdf[name] = np.histogram(sx[ok], bins=x_edges, density=True)[0] if ok.any() else np.zeros(bins)
        y_pdf[name] = np.histogram(sy[ok], bins=y_edges, density=True)[0] if ok.any() else np.zeros(bins)

    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("white")

    fig = plt.figure(figsize=(13.5, 12.0))
    outer = fig.add_gridspec(
        2, 2, hspace=0.20, wspace=0.18, left=0.06, right=0.94, top=0.93, bottom=0.05
    )

    last_im = None
    for name in PANEL_ORDER:
        i, j = PANEL_GRID_POSITIONS[name]
        inner = outer[i, j].subgridspec(
            2, 2,
            width_ratios=[1, 4],
            height_ratios=[4, 1],
            hspace=0.04,
            wspace=0.04,
        )
        ax_y = fig.add_subplot(inner[0, 0])
        ax_main = fig.add_subplot(inner[0, 1], sharey=ax_y)
        ax_x = fig.add_subplot(inner[1, 1], sharex=ax_main)

        sx = _field_values(samples[name], x_field)
        sy = _field_values(samples[name], y_field)
        ok = np.isfinite(sx) & np.isfinite(sy)
        sx = sx[ok]
        sy = sy[ok]
        n = int(sx.size)
        if n:
            h, _, _ = np.histogram2d(sx, sy, bins=[x_edges, y_edges])
            h = h / max(1.0, h.sum())
            with np.errstate(divide="ignore"):
                shown = np.log10(np.where(h > 0, h, np.nan))
            im = ax_main.pcolormesh(x_edges, y_edges, shown.T, cmap=cmap, shading="auto")
            last_im = im
            if contour is not None:
                xc, yc, field, levels = contour
                try:
                    ax_main.contour(xc, yc, field, levels=levels, colors="black", linewidths=1.2)
                except ValueError:
                    pass
        ax_main.set_facecolor("white")
        ax_main.set_xlim(x_lo, x_hi)
        ax_main.set_ylim(y_lo, y_hi)
        ax_main.set_title(f"{label_for_name[name]} (n={n:,})")
        ax_main.grid(True, alpha=0.3)
        plt.setp(ax_main.get_xticklabels(), visible=False)
        plt.setp(ax_main.get_yticklabels(), visible=False)

        for ds_name in PANEL_ORDER:
            color = color_for_name[ds_name]
            lw = 2.4 if ds_name == name else 1.0
            alpha = 1.0 if ds_name == name else 0.7
            zorder = 5 if ds_name == name else 2
            ax_x.plot(x_centers, x_pdf[ds_name], color=color, linewidth=lw, alpha=alpha, zorder=zorder)
            ax_y.plot(y_pdf[ds_name], y_centers, color=color, linewidth=lw, alpha=alpha, zorder=zorder)
        ax_x.set_xlim(x_lo, x_hi)
        ax_x.set_ylim(bottom=0.0)
        ax_x.grid(True, alpha=0.3)
        ax_x.set_xlabel(x_label)
        ax_x.set_ylabel("PDF")
        ax_y.set_ylim(y_lo, y_hi)
        ax_y.set_xlim(left=0.0)
        ax_y.invert_xaxis()
        ax_y.grid(True, alpha=0.3)
        ax_y.set_ylabel(y_label)
        ax_y.set_xlabel("PDF")

    legend_handles = [
        plt.Line2D([], [], color=color_for_name[n], lw=2.0, label=label_for_name[n])
        for n in PANEL_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
    )
    if last_im is not None:
        cbar_ax = fig.add_axes([0.96, 0.10, 0.012, 0.78])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("log$_{10}$ frequency")
        cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    if title:
        fig.suptitle(title, y=0.965)
    fig.savefig(out_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create per-cell DSD plots with normalized-gamma obs and models.",
    )
    parser.add_argument(
        "--samples-npz",
        type=Path,
        default=PROCESSED_DIR / "disdrometer_dsd_percell_samples_all_leads.npz",
        help="Existing per-cell samples NPZ produced by run_dsd_per_cell.py.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=FIGURE_DIR / "normalized_gamma_percell",
        help="Output folder for the new plot set.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Folder for the small normalized-gamma obs diagnostics NPZ.",
    )
    parser.add_argument("--mu-min", type=float, default=-0.95)
    parser.add_argument("--mu-max", type=float, default=50.0)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--skip-fitted", action="store_true")
    parser.add_argument("--skip-mu1", action="store_true")
    parser.add_argument("--skip-mu1-lwc-nt", action="store_true")
    parser.add_argument("--skip-diagnostics", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading per-cell samples from {args.samples_npz}", flush=True)
    with np.load(args.samples_npz) as npz:
        obs_empirical = _load_prefixed(npz, "obs")
        samples = {
            "control": _load_prefixed(npz, "control"),
            "graupel": _load_prefixed(npz, "graupel"),
            "2mom": _load_prefixed(npz, "2mom"),
        }

    obs_gamma = normalized_gamma_from_empirical_samples(
        obs_empirical,
        mu_min=args.mu_min,
        mu_max=args.mu_max,
    )
    samples_native = {
        "obs": obs_gamma,
        "control": _with_constant_mu(samples["control"], 0.0),
        "graupel": _with_constant_mu(samples["graupel"], 0.0),
        "2mom": _with_constant_mu(samples["2mom"], 1.0),
    }

    out_npz = args.processed_dir / "disdrometer_dsd_obs_normalized_gamma_all_leads.npz"
    payload = {f"obs_gamma__{key}": value for key, value in obs_gamma.items()}
    np.savez_compressed(out_npz, **payload)
    print(f"normalized-gamma obs diagnostics -> {out_npz}", flush=True)
    print(
        "sample counts: "
        + ", ".join(
            f"{name}={_finite_count(samples_native[name]):,}"
            for name in ("obs", *EXPERIMENTS)
        ),
        flush=True,
    )
    if obs_gamma["mu"].size:
        p = np.nanpercentile(obs_gamma["mu"], [5, 50, 95])
        print(
            f"obs gamma mu percentiles: p05={p[0]:.2f}, p50={p[1]:.2f}, p95={p[2]:.2f}",
            flush=True,
        )

    if not args.skip_fitted:
        for x_field, x_label, suffix in (
            ("d0_mm", "D$_0$ (mm)", "logNw_D0"),
            ("dm_mm", "D$_m$ (mm)", "logNw_Dm"),
        ):
            out = args.figure_dir / f"dsd_percell_normalized_gamma_{suffix}_all_leads.png"
            title = (
                f"log$_{{10}}$ N$_w$ vs {'D$_0$' if x_field == 'd0_mm' else 'D$_m$'}, "
                "normalized gamma, per-cell"
            )
            plot_2x2_with_marginals(
                out_path=out,
                samples=samples_native,
                x_field=x_field,
                x_label=x_label,
                title=title,
                bins=args.bins,
            )
            print(f"rendered {out}", flush=True)

    samples_mu1: dict[str, dict[str, np.ndarray]] | None = None
    if not args.skip_mu1 or not args.skip_diagnostics:
        print("projecting obs, C1M, G1M, and G2M to normalized gamma with mu=1", flush=True)
        samples_mu1 = {
            "obs": _normalized_gamma_fixed_mu(obs_empirical, 1.0),
            "control": _normalized_gamma_fixed_mu(samples["control"], 1.0),
            "graupel": _normalized_gamma_fixed_mu(samples["graupel"], 1.0),
            "2mom": _normalized_gamma_fixed_mu(samples["2mom"], 1.0),
        }
        out_npz_mu1 = args.processed_dir / "disdrometer_dsd_percell_mu1_all_leads.npz"
        np.savez_compressed(
            out_npz_mu1,
            **{
                f"{name}__{key}": value
                for name, values in samples_mu1.items()
                for key, value in values.items()
            },
        )
        print(f"mu=1 diagnostics -> {out_npz_mu1}", flush=True)

    if not args.skip_mu1 and samples_mu1 is not None:
        mu1_dir = args.figure_dir.parent / "normalized_gamma_percell_mu1"
        mu1_dir.mkdir(parents=True, exist_ok=True)
        for x_field, x_label, suffix in (
            ("d0_mm", "D$_0$ (mm)", "logNw_D0"),
            ("dm_mm", "D$_m$ (mm)", "logNw_Dm"),
        ):
            out = mu1_dir / f"dsd_percell_normalized_gamma_mu1_{suffix}_all_leads.png"
            title = (
                f"log$_{{10}}$ N$_w$ vs {'D$_0$' if x_field == 'd0_mm' else 'D$_m$'}, "
                "normalized gamma, per-cell, mu=1"
            )
            plot_2x2_with_marginals(
                out_path=out,
                samples=samples_mu1,
                x_field=x_field,
                x_label=x_label,
                title=title,
                bins=args.bins,
            )
            print(f"rendered {out}", flush=True)

    samples_mu1_lwc_nt: dict[str, dict[str, np.ndarray]] | None = None
    if not args.skip_mu1_lwc_nt or not args.skip_diagnostics:
        print("projecting obs, C1M, G1M, and G2M to mu=1 fitted from LWC + Nt", flush=True)
        samples_mu1_lwc_nt = {
            "obs": _normalized_gamma_fixed_mu_from_lwc_nt(obs_empirical, 1.0),
            "control": _normalized_gamma_fixed_mu_from_lwc_nt(samples["control"], 1.0),
            "graupel": _normalized_gamma_fixed_mu_from_lwc_nt(samples["graupel"], 1.0),
            "2mom": _normalized_gamma_fixed_mu_from_lwc_nt(samples["2mom"], 1.0),
        }
        out_npz_mu1_lwc_nt = args.processed_dir / "disdrometer_dsd_percell_mu1_lwc_nt_all_leads.npz"
        np.savez_compressed(
            out_npz_mu1_lwc_nt,
            **{
                f"{name}__{key}": value
                for name, values in samples_mu1_lwc_nt.items()
                for key, value in values.items()
            },
        )
        print(f"mu=1 LWC+Nt diagnostics -> {out_npz_mu1_lwc_nt}", flush=True)

    if not args.skip_mu1_lwc_nt and samples_mu1_lwc_nt is not None:
        mu1_lwc_nt_dir = args.figure_dir.parent / "normalized_gamma_percell_mu1_lwc_nt"
        mu1_lwc_nt_dir.mkdir(parents=True, exist_ok=True)
        for x_field, x_label, suffix in (
            ("d0_mm", "D$_0$ (mm)", "logNw_D0"),
            ("dm_mm", "D$_m$ (mm)", "logNw_Dm"),
        ):
            out = mu1_lwc_nt_dir / f"dsd_percell_normalized_gamma_mu1_lwc_nt_{suffix}_all_leads.png"
            title = (
                f"log$_{{10}}$ N$_w$ vs {'D$_0$' if x_field == 'd0_mm' else 'D$_m$'}, "
                "normalized gamma, per-cell, mu=1 from LWC+Nt"
            )
            plot_2x2_with_marginals(
                out_path=out,
                samples=samples_mu1_lwc_nt,
                x_field=x_field,
                x_label=x_label,
                title=title,
                bins=args.bins,
            )
            print(f"rendered {out}", flush=True)

    if not args.skip_diagnostics:
        diagnostics_dir = args.figure_dir.parent / "normalized_gamma_percell_diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        diagnostics = (
            ("dm_mm", "mu", "D$_m$ (mm)", "$\\mu$", "mu_vs_Dm"),
            ("dm_mm", "d0_over_dm", "D$_m$ (mm)", "D$_0$ / D$_m$", "D0_over_Dm_vs_Dm"),
            (
                "dm_mm",
                "sigma_over_dm",
                "D$_m$ (mm)",
                "$\\sigma_m$ / D$_m$",
                "sigma_over_Dm_vs_Dm",
            ),
        )
        for sample_tag, sample_set, label in (
            ("native_fitted", samples_native, "native/fitted mu"),
            ("mu1_all", samples_mu1, "mu=1 for all"),
            ("mu1_lwc_nt", samples_mu1_lwc_nt, "mu=1 from LWC+Nt"),
        ):
            if sample_set is None:
                continue
            for x_field, y_field, x_label, y_label, suffix in diagnostics:
                out = diagnostics_dir / f"dsd_percell_{sample_tag}_{suffix}_all_leads.png"
                _plot_2x2_generic_with_marginals(
                    out_path=out,
                    samples=sample_set,
                    x_field=x_field,
                    y_field=y_field,
                    x_label=x_label,
                    y_label=y_label,
                    title=f"{y_label} vs D$_m$, {label}, per-cell",
                    bins=args.bins,
                    min_x=0.2,
                )
                print(f"rendered {out}", flush=True)


if __name__ == "__main__":
    main()
