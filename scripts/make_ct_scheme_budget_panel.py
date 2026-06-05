#!/usr/bin/env python3
"""Plot focused CT two-year mean budget terms with the proper CT FBL split."""

import argparse
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RAW_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-0024-only")
CACHE_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-processed/DDH-0024-only-CT-extracted"
)
OUTPUT_DIR = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/paper1/10_ct_scheme_budget"
)
TOOLBOX = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ddhtoolbox")
RUNTIME_ROOT = Path(
    "/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/DDH-processed/lead0024_VZ/_runtime"
)
TEMP_ROOT = Path("/mnt/HDS_CLIMATE/CLIMATE/deba/ALARO-RUNS/processed-data")
LEAD = "0024"
YCOOR = "VZ"
VARIABLE = "CT"
FREEZING_K = 273.15
EXPERIMENTS = ("control", "graupel", "2mom")
EXP_LABELS = {"control": "C1M", "graupel": "G1M", "2mom": "G2M"}
COMPONENTS_BY_EXPERIMENT = {
    "control": ("micro-cv", "micro-rs"),
    "graupel": ("micro-cv", "micro-rs"),
    "2mom": ("micro-cv", "micro-rs"),
}
COMPONENT_LABELS = {
    "micro-cv": "Convection scheme 3MT",
    "micro-rs": "Microphysics scheme",
}
COMPONENT_STYLES = {
    "micro-cv": {"color": "#d95f02", "linestyle": "--"},
    "micro-rs": {"color": "#7570b3", "linestyle": "-"},
}
FIGURE_NAME = "ct_scheme_budget_450dpi.png"
TEXT_NAME = "ct_scheme_budget_data.txt"
RUNTIME_VERSION = "ct_alaro_2ice_micro_terms_v1"
FBL_FILE_BY_EXPERIMENT = {
    "control": Path("alaro") / "CT3.fbl-2ice",
    "graupel": Path("alaro") / "CT.fbl-3ice",
    "2mom": Path("alaro") / "CT.fbl-3ice",
}
FBL_TAG_BY_EXPERIMENT = {
    "control": "CT3_fbl_2ice_micro",
    "graupel": "CT_fbl_3ice",
    "2mom": "CT_fbl_3ice",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Rerender CT_2yr_avg_budget with the correct 2-ice/3-ice CT FBL terms."
        )
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--toolbox", type=Path, default=TOOLBOX)
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME_ROOT)
    parser.add_argument("--lead", default=LEAD)
    parser.add_argument("--ycoor", default=YCOOR, choices=("VP", "VZ"))
    parser.add_argument(
        "--max-altitude-km",
        type=float,
        default=20.0,
        help="Only used when --ycoor VZ.",
    )
    parser.add_argument("--extract-workers", type=int, default=24)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--limit-days", type=int, default=None)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args(argv)


def raw_files(raw_root, experiment, lead):
    output_dir = raw_root / experiment / "output"
    if not output_dir.exists():
        return []
    pattern = "DDH*_DHFDLABOF+{}".format(lead)
    return sorted(path for path in output_dir.glob(pattern) if path.is_file())


def day_from_raw(path):
    return path.name.split("_", 1)[0]


def prepare_scheme_runtime(cache_dir, toolbox, runtime_root):
    scheme_root = cache_dir / "_runtime_real_alaro_fbl_split"
    marker = scheme_root / "VERSION"
    if marker.exists() and marker.read_text(encoding="ascii").strip() == RUNTIME_VERSION:
        return scheme_root

    if scheme_root.exists():
        shutil.rmtree(str(scheme_root))

    for experiment in EXPERIMENTS:
        src = runtime_root / experiment / "ddh_budget_lists"
        dst = scheme_root / experiment / "ddh_budget_lists"
        shutil.copytree(str(src), str(dst))
        shutil.copy2(
            str(toolbox / "ddh_budget_lists" / FBL_FILE_BY_EXPERIMENT[experiment]),
            str(dst / "alaro" / "CT.fbl"),
        )

    marker.write_text(RUNTIME_VERSION + "\n", encoding="ascii")
    return scheme_root


def read_dta(path):
    data = np.loadtxt(str(path))
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Expected two-column .dta file: {}".format(path))
    return data[:, 0].astype(np.float64), data[:, 1].astype(np.float64)


def load_height_axis_km(temp_root):
    path = temp_root / "geopotential" / "2years" / "control_full-domain_height_profile_first.npz"
    with np.load(str(path)) as data:
        return np.asarray(data["height_m"], dtype=np.float64) / 1000.0


def load_temperature_mean(temp_root, experiment):
    path = temp_root / "temperature" / "2years" / "{}_full-domain_diurnal_profile.npz".format(experiment)
    with np.load(str(path)) as data:
        return np.asarray(data["mean"], dtype=np.float64)


def compute_freezing_line_km(height_km, temperature):
    n_levels = min(height_km.size, temperature.shape[0])
    y = np.asarray(height_km[:n_levels], dtype=np.float64)
    temp = np.asarray(temperature[:n_levels, :], dtype=np.float64)
    order = np.argsort(y)
    y = y[order]
    temp = temp[order, :]
    out = np.full(24, np.nan, dtype=np.float64)
    for hour in range(24):
        column = temp[:, hour]
        finite = np.isfinite(y) & np.isfinite(column)
        if np.sum(finite) < 2:
            continue
        yy = y[finite]
        tt = column[finite]
        for idx in range(yy.size - 1):
            t1 = tt[idx]
            t2 = tt[idx + 1]
            y1 = yy[idx]
            y2 = yy[idx + 1]
            if np.isclose(t1, FREEZING_K):
                out[hour] = y1
                break
            if np.isclose(t2, FREEZING_K):
                out[hour] = y2
                break
            if (t1 - FREEZING_K) * (t2 - FREEZING_K) < 0.0 and not np.isclose(t1, t2):
                frac = (FREEZING_K - t1) / (t2 - t1)
                out[hour] = y1 + frac * (y2 - y1)
                break
    return out


def freezing_mean_by_experiment(temp_root):
    height = load_height_axis_km(temp_root)
    out = {}
    for experiment in EXPERIMENTS:
        temperature = load_temperature_mean(temp_root, experiment)
        line = compute_freezing_line_km(height, temperature)
        out[experiment] = float(np.nanmean(line))
    return out


def extracted_ct_dir(cache_dir, lead, ycoor, experiment, day):
    tag = FBL_TAG_BY_EXPERIMENT[experiment]
    return cache_dir / "lead{}_{}_{}".format(lead, ycoor, tag) / experiment / day / VARIABLE


def extracted_component_path(cache_dir, lead, ycoor, experiment, day, component):
    return extracted_ct_dir(cache_dir, lead, ycoor, experiment, day) / (
        "{}.DHFDLABOF+{}.{}.dta".format(VARIABLE, lead, component)
    )


def runtime_bps(runtime_root, experiment):
    return runtime_root / experiment / "ddh_budget_lists"


def extraction_is_complete(out_dir, lead, required_components):
    if not (out_dir / "done.ok").exists():
        return False
    for component in required_components:
        path = out_dir / "{}.DHFDLABOF+{}.{}.dta".format(VARIABLE, lead, component)
        if not path.exists():
            return False
    return True


def extract_ct_day(task):
    raw_path, raw_root, cache_dir, toolbox, runtime_root, experiment, lead, ycoor, force = task
    day = day_from_raw(raw_path)
    out_dir = extracted_ct_dir(cache_dir, lead, ycoor, experiment, day)
    log_path = out_dir / "extract.log"
    required_components = COMPONENTS_BY_EXPERIMENT[experiment]
    if extraction_is_complete(out_dir, lead, required_components) and not force:
        return experiment, day, "skip", ""

    bps = runtime_bps(runtime_root, experiment)
    conversion_list = bps / "alaro" / "conversion_list"
    if not conversion_list.exists():
        return experiment, day, "fail", "missing runtime conversion_list: {}".format(conversion_list)

    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / "done.ok"
    if done_path.exists():
        done_path.unlink()
    with tempfile.TemporaryDirectory(prefix="ct_input_") as input_dir_name:
        with tempfile.TemporaryDirectory(prefix="ct_work_") as work_dir_name:
            input_dir = Path(input_dir_name)
            work_dir = Path(work_dir_name)
            input_name = "DHFDLABOF+{}".format(lead)
            os.symlink(str(raw_path), str(input_dir / input_name))
            env = os.environ.copy()
            env["DDHTOOLBOX"] = str(toolbox)
            env["DDHB_BPS"] = str(bps)
            env["DDHI_LIST"] = str(conversion_list)
            env.pop("DDH_PLOT", None)
            env["PATH"] = "{}:{}:{}:{}".format(
                toolbox / "tools",
                toolbox / "tools" / "lfa",
                toolbox / "tools" / ".dd2gr" / "src",
                env.get("PATH", ""),
            )
            cmd = [
                "ddhb",
                "-v",
                "alaro/{}".format(VARIABLE),
                "-i",
                input_name,
                "-Y",
                ycoor,
                "-r",
                str(work_dir),
            ]
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(
                    cmd,
                    cwd=str(input_dir),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            if proc.returncode != 0:
                return experiment, day, "fail", "ddhb return code {}".format(proc.returncode)

            budget_dir = work_dir / "budget.alaro"
            if not budget_dir.exists():
                return experiment, day, "fail", "missing budget.alaro output"

            for old_file in out_dir.glob("*.dta"):
                old_file.unlink()
            for dta in budget_dir.glob("*.dta"):
                shutil.copy2(str(dta), str(out_dir / dta.name))
            missing_required = []
            for component in required_components:
                required_path = out_dir / "{}.DHFDLABOF+{}.{}.dta".format(
                    VARIABLE, lead, component
                )
                if not required_path.exists():
                    missing_required.append(component)
            if missing_required:
                return (
                    experiment,
                    day,
                    "fail",
                    "missing required CT terms: {}".format(", ".join(missing_required)),
                )
            done_path.write_text("ok\n", encoding="ascii")
    return experiment, day, "ok", ""


def extract_all_ct(
    raw_root,
    cache_dir,
    toolbox,
    runtime_root,
    lead,
    ycoor,
    workers,
    force,
    experiments,
    limit_days,
):
    tasks = []
    for experiment in experiments:
        files = raw_files(raw_root, experiment, lead)
        if limit_days is not None:
            files = files[: max(0, limit_days)]
        for raw_path in files:
            tasks.append(
                (
                    raw_path,
                    raw_root,
                    cache_dir,
                    toolbox,
                    runtime_root,
                    experiment,
                    lead,
                    ycoor,
                    force,
                )
            )
    if not tasks:
        raise RuntimeError("No raw +{} DDH files found under {}".format(lead, raw_root))

    workers = max(1, min(int(workers), 64))
    counts = {"ok": 0, "skip": 0, "fail": 0}
    failures = []
    print("Extracting CT from {} raw files with {} workers...".format(len(tasks), workers), flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(extract_ct_day, task): task for task in tasks}
        for future in as_completed(future_map):
            experiment, day, status, note = future.result()
            counts[status] = counts.get(status, 0) + 1
            done += 1
            if status == "fail":
                failures.append((experiment, day, note))
            if done % 100 == 0 or done == len(tasks):
                print(
                    "  extracted {}/{} (ok={}, skip={}, fail={})".format(
                        done, len(tasks), counts.get("ok", 0), counts.get("skip", 0), counts.get("fail", 0)
                    ),
                    flush=True,
                )
    if failures:
        for experiment, day, note in failures[:10]:
            print("  FAIL {} {}: {}".format(experiment, day, note), flush=True)
        raise RuntimeError("{} CT extraction tasks failed".format(len(failures)))
    return counts


def average_component(raw_root, cache_dir, experiment, component, lead, ycoor):
    raw = raw_files(raw_root, experiment, lead)
    prefix = "{}.DHFDLABOF+{}.".format(VARIABLE, lead)
    profiles = []
    altitude = None
    missing = 0
    bad = 0

    for raw_path in raw:
        day = day_from_raw(raw_path)
        path = extracted_component_path(cache_dir, lead, ycoor, experiment, day, component)
        if not path.exists():
            missing += 1
            continue
        try:
            z, values = read_dta(path)
        except Exception:
            bad += 1
            continue
        if altitude is None:
            altitude = z
        elif z.shape != altitude.shape:
            bad += 1
            continue
        profiles.append(values)

    if not profiles:
        return {
            "coord": None,
            "mean": None,
            "n_days_total": len(raw),
            "n_days_used": 0,
            "missing": missing,
            "bad": bad,
        }

    stack = np.vstack(profiles)
    return {
        "coord": altitude,
        "mean": np.nanmean(stack, axis=0),
        "n_days_total": len(raw),
        "n_days_used": stack.shape[0],
        "missing": missing,
        "bad": bad,
    }


def collect_data(raw_root, cache_dir, lead, ycoor):
    out = {}
    for experiment in EXPERIMENTS:
        for component in COMPONENTS_BY_EXPERIMENT[experiment]:
            out[(experiment, component)] = average_component(
                raw_root, cache_dir, experiment, component, lead, ycoor
            )
    return out


def plot_coordinate(coord, ycoor):
    y = np.asarray(coord, dtype=np.float64)
    if ycoor == "VP":
        y = -y
    return y


def sorted_for_plot(coord, values, ycoor, max_altitude_km):
    z = plot_coordinate(coord, ycoor)
    v = np.asarray(values, dtype=np.float64)
    order = np.argsort(z)
    z = z[order]
    v = v[order]
    if ycoor == "VZ":
        keep = np.isfinite(z) & (z >= 0.0) & (z <= max_altitude_km)
    else:
        keep = np.isfinite(z) & (z >= 0.0)
    return z[keep], v[keep]


def finite_xrange(data, ycoor, max_altitude_km, experiment=None):
    values = []
    for (item_experiment, _component), result in data.items():
        if experiment is not None and item_experiment != experiment:
            continue
        if result["coord"] is None or result["mean"] is None:
            continue
        _, profile = sorted_for_plot(
            result["coord"], result["mean"], ycoor, max_altitude_km
        )
        profile = profile[np.isfinite(profile)]
        if profile.size:
            values.append(profile)
    if not values:
        return -1.0, 1.0
    merged = np.concatenate(values)
    xmin = float(np.nanmin(merged))
    xmax = float(np.nanmax(merged))
    if xmin == xmax:
        pad = max(abs(xmin) * 0.1, 0.1)
    else:
        pad = 0.08 * (xmax - xmin)
    return xmin - pad, xmax + pad


def availability_note(data):
    parts = []
    complete = True
    for experiment in EXPERIMENTS:
        used = []
        total = []
        for component in COMPONENTS_BY_EXPERIMENT[experiment]:
            result = data[(experiment, component)]
            used.append(result["n_days_used"])
            total.append(result["n_days_total"])
            if result["n_days_used"] != result["n_days_total"]:
                complete = False
        parts.append(
            "{} {}-{} / {}".format(
                EXP_LABELS[experiment],
                min(used),
                max(used),
                max(total) if total else 0,
            )
        )
    if complete:
        return "2-year mean from all available DDH days"
    return "available terms only (" + "; ".join(parts) + ")"


def plot(data, output_path, ycoor, max_altitude_km, dpi, freezing_means=None):
    note = availability_note(data)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.6), sharex=False, sharey=True)

    for ax, experiment in zip(axes, EXPERIMENTS):
        ax.set_title(EXP_LABELS[experiment], fontsize=18, fontweight="bold")
        plotted = 0
        for component in COMPONENTS_BY_EXPERIMENT[experiment]:
            result = data[(experiment, component)]
            if result["coord"] is None or result["mean"] is None:
                continue
            z, values = sorted_for_plot(
                result["coord"], result["mean"], ycoor, max_altitude_km
            )
            style = COMPONENT_STYLES[component]
            ax.plot(
                values,
                z,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.8,
                label=COMPONENT_LABELS[component],
            )
            plotted += 1

        if plotted == 0:
            ax.text(
                0.5,
                0.5,
                "No selected CT terms\navailable",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=13,
                color="0.35",
            )

        ax.axvline(0.0, color="0.35", linewidth=0.9)
        if ycoor == "VZ" and freezing_means and experiment in freezing_means:
            ax.axhline(
                freezing_means[experiment],
                color="black",
                linestyle=":",
                linewidth=2.0,
                label="0 C isotherm",
            )
        ax.grid(True, alpha=0.28)
        xmin, xmax = finite_xrange(data, ycoor, max_altitude_km, experiment=experiment)
        ax.set_xlim(xmin, xmax)
        if ycoor == "VP":
            ax.set_ylim(1050.0, 0.0)
        else:
            ax.set_ylim(0.0, max_altitude_km)
        ax.set_xlabel("K day$^{-1}$", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)

    ylabel = "Pressure (hPa)" if ycoor == "VP" else "Altitude (km)"
    axes[0].set_ylabel(ylabel, fontsize=14)

    handles = []
    labels = []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles.extend(ax_handles)
        labels.extend(ax_labels)
    if handles:
        unique = {}
        unique_handles = []
        unique_labels = []
        for handle, label in zip(handles, labels):
            if label in unique:
                continue
            unique[label] = True
            unique_handles.append(handle)
            unique_labels.append(label)
        fig.legend(
            unique_handles,
            unique_labels,
            loc="lower center",
            ncol=3,
            frameon=True,
            framealpha=0.95,
            fontsize=12,
            bbox_to_anchor=(0.5, 0.02),
        )

    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_text(data, path, raw_root, cache_dir, lead, ycoor, max_altitude_km, freezing_means=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("CT +24 h budget: focused scheme panel\n")
        f.write("Source script originally found: alaro_analysis/ddh/scripts/yearly_average_budgets.py\n")
        f.write("Raw source root: {}\n".format(raw_root))
        f.write("Extracted CT cache: {}\n".format(cache_dir))
        f.write("Lead: +{}\n".format(lead))
        if ycoor == "VP":
            f.write("Vertical coordinate: pressure from VP .dta files, plotted as positive hPa\n")
        else:
            f.write("Vertical coordinate: altitude from VZ .dta files\n")
        f.write("Availability note: {}\n".format(availability_note(data)))
        f.write("FBL choice by experiment:\n")
        for experiment in EXPERIMENTS:
            f.write(
                "  {} / {}: {}\n".format(
                    experiment,
                    EXP_LABELS[experiment],
                    FBL_FILE_BY_EXPERIMENT[experiment],
                )
            )
        f.write("Included raw components by experiment:\n")
        for experiment in EXPERIMENTS:
            f.write("  {} / {}:\n".format(experiment, EXP_LABELS[experiment]))
            for component in COMPONENTS_BY_EXPERIMENT[experiment]:
                f.write("    {} -> {}\n".format(component, COMPONENT_LABELS[component]))
        if ycoor == "VZ":
            f.write("Maximum plotted altitude: {} km\n\n".format(max_altitude_km))
            if freezing_means:
                f.write("0 C isotherm mean height by experiment\n")
                for experiment in EXPERIMENTS:
                    f.write(
                        "{},{},{:.12g}\n".format(
                            experiment,
                            EXP_LABELS[experiment],
                            freezing_means[experiment],
                        )
                    )
                f.write("\n")
        else:
            f.write("Pressure axis plotted from 1050 to 0 hPa\n\n")

        f.write("Availability\n")
        f.write("experiment,label,component,display_label,n_days_total,n_days_used,missing,bad\n")
        for experiment in EXPERIMENTS:
            for component in COMPONENTS_BY_EXPERIMENT[experiment]:
                result = data[(experiment, component)]
                f.write(
                    "{},{},{},{},{},{},{},{}\n".format(
                        experiment,
                        EXP_LABELS[experiment],
                        component,
                        COMPONENT_LABELS[component],
                        result["n_days_total"],
                        result["n_days_used"],
                        result["missing"],
                        result["bad"],
                    )
                )

        f.write("\nProfile data\n")
        coord_name = "pressure_hpa" if ycoor == "VP" else "altitude_km"
        f.write("experiment,label,component,display_label,{},mean_k_day\n".format(coord_name))
        for experiment in EXPERIMENTS:
            for component in COMPONENTS_BY_EXPERIMENT[experiment]:
                result = data[(experiment, component)]
                if result["coord"] is None or result["mean"] is None:
                    continue
                z, values = sorted_for_plot(
                    result["coord"], result["mean"], ycoor, max_altitude_km
                )
                for zi, vi in zip(z, values):
                    f.write(
                        "{},{},{},{},{:.12g},{:.12g}\n".format(
                            experiment,
                            EXP_LABELS[experiment],
                            component,
                            COMPONENT_LABELS[component],
                            zi,
                            vi,
                        )
                    )


def main(argv=None):
    args = parse_args(argv)
    raw_root = args.raw_root.resolve()
    cache_dir = args.cache_dir.resolve()
    toolbox = args.toolbox.resolve()
    runtime_root = args.runtime_root.resolve()
    output_dir = args.output_dir.resolve()
    scheme_runtime_root = prepare_scheme_runtime(cache_dir, toolbox, runtime_root)
    extract_all_ct(
        raw_root,
        cache_dir,
        toolbox,
        scheme_runtime_root,
        args.lead,
        args.ycoor,
        args.extract_workers,
        args.force_extract,
        args.experiments,
        args.limit_days,
    )
    data = collect_data(raw_root, cache_dir, args.lead, args.ycoor)
    freezing_means = freezing_mean_by_experiment(TEMP_ROOT) if args.ycoor == "VZ" else None
    figure_path = output_dir / FIGURE_NAME
    text_path = output_dir / "data_txt" / TEXT_NAME
    plot(data, figure_path, args.ycoor, args.max_altitude_km, args.dpi, freezing_means)
    write_text(data, text_path, raw_root, cache_dir, args.lead, args.ycoor, args.max_altitude_km, freezing_means)
    print("[saved] {}".format(figure_path))
    print("[saved] {}".format(text_path))
    for experiment in EXPERIMENTS:
        counts = []
        for component in COMPONENTS_BY_EXPERIMENT[experiment]:
            result = data[(experiment, component)]
            counts.append("{}={}/{}".format(component, result["n_days_used"], result["n_days_total"]))
        print("{}: {}".format(EXP_LABELS[experiment], ", ".join(counts)))


if __name__ == "__main__":
    main()
