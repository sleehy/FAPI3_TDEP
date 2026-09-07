#!/usr/bin/env python3
"""Plot a selected frequency window from completed TDEP dispersion calculations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_config(path: Path) -> dict:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The configuration must be a YAML mapping.")
    config["_config_dir"] = path.parent.resolve()
    return config


def config_path(config: dict, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config["_config_dir"] / path


def available_iterations(output_dir: Path) -> dict[int, Path]:
    """Return completed iteration directories that contain saved phonon bands."""
    iterations = {}
    for directory in output_dir.glob("iteration_*"):
        if not directory.is_dir():
            continue
        try:
            iteration = int(directory.name.removeprefix("iteration_"))
        except ValueError:
            continue
        if (directory / "dispersion_relations_THz.npy").is_file() or (directory / "outfile.dispersion_relations").is_file():
            iterations[iteration] = directory
    return dict(sorted(iterations.items()))


def default_iterations(available: dict[int, Path], config: dict) -> list[int]:
    """Mirror the main overlay's iteration selection, restricted to available data."""
    dispersion = config["dispersion"]
    start = int(dispersion.get("overlay_start_iteration", 1))
    interval = int(dispersion.get("overlay_interval", 3))
    if start < 1 or interval < 1:
        raise ValueError("dispersion.overlay_start_iteration and overlay_interval must be positive integers.")
    return [iteration for iteration in available if iteration >= start and (iteration - start) % interval == 0]


def load_bands(iteration_dir: Path) -> np.ndarray:
    filename = iteration_dir / "dispersion_relations_THz.npy"
    bands = np.load(filename) if filename.is_file() else np.loadtxt(iteration_dir / "outfile.dispersion_relations")
    bands = np.atleast_2d(bands)
    if bands.shape[1] < 2:
        raise ValueError(f"Unexpected dispersion data in {iteration_dir}.")
    return bands


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("tdep_tetragonal.yaml"), help="TDEP YAML configuration file.")
    parser.add_argument("--output-dir", type=Path, help="Completed TDEP output directory (defaults to output.directory in --config).")
    parser.add_argument("--frequency-min", required=True, type=float, help="Lower y-axis bound in THz.")
    parser.add_argument("--frequency-max", required=True, type=float, help="Upper y-axis bound in THz.")
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        help="Iteration numbers to overlay. Defaults to the main plot's overlay selection.",
    )
    parser.add_argument("--output", type=Path, help="Output PNG path (default: in the TDEP output directory).")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.frequency_min >= args.frequency_max:
        raise ValueError("--frequency-min must be smaller than --frequency-max.")
    config = load_config(args.config)
    output_dir = args.output_dir.resolve() if args.output_dir else config_path(config, config["output"]["directory"])
    available = available_iterations(output_dir)
    if not available:
        raise FileNotFoundError(f"No saved dispersion data found in {output_dir}.")
    iterations = args.iterations if args.iterations is not None else default_iterations(available, config)
    missing = [iteration for iteration in iterations if iteration not in available]
    if missing:
        raise FileNotFoundError(f"No saved dispersion data for iteration(s): {missing}.")
    if not iterations:
        raise ValueError("No iterations were selected for the overlay.")

    all_bands = [(iteration, load_bands(available[iteration])) for iteration in iterations]
    reference_x = all_bands[-1][1][:, 0]
    for iteration, bands in all_bands[:-1]:
        if bands.shape[0] != len(reference_x) or not np.allclose(bands[:, 0], reference_x):
            raise ValueError(f"Iteration {iteration} has a different q-point path and cannot be overlaid.")

    labels = config["dispersion"]["labels"]
    points_per_segment = int(config["dispersion"]["points_per_segment"])
    tick_indices = [min(index * points_per_segment, len(reference_x) - 1) for index in range(len(labels))]
    ticks = reference_x[tick_indices]
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(all_bands)))
    for (iteration, bands), color in zip(all_bands, colors):
        ax.plot(bands[:, 0], bands[:, 1:], color=color, alpha=0.8, lw=0.6)
        ax.plot([], [], color=color, lw=2, label=f"iteration {iteration}")
    for boundary in ticks:
        ax.axvline(boundary, color="0.75", lw=0.7)
    if args.frequency_min <= 0.0 <= args.frequency_max:
        ax.axhline(0.0, color="0.2", lw=0.8)
    ax.set(
        xlim=(reference_x[0], reference_x[-1]),
        ylim=(args.frequency_min, args.frequency_max),
        xticks=ticks,
        xticklabels=labels,
        ylabel="Frequency (THz)",
        title=(
            f"TDEP phonon dispersion zoom: {args.frequency_min:g}–{args.frequency_max:g} THz "
            f"at {config['tdep']['temperature_K']:g} K"
        ),
    )
    ax.legend(ncol=2, fontsize=8, frameon=False)
    output = args.output if args.output else output_dir / (
        f"phonon_dispersion_zoom_{args.frequency_min:g}-{args.frequency_max:g}_THz.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240)
    plt.close(fig)
    print(f"Zoomed dispersion plot: {output}")
    print("Iterations: " + ", ".join(map(str, iterations)))


if __name__ == "__main__":
    main()
