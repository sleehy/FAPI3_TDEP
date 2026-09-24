#!/usr/bin/env python3
"""Plot low-frequency phonon-band linewidth ratios, Gamma / omega.

The script reads the ``frequency`` and ``gamma`` datasets in a phono3py kappa
HDF5 file.  It produces a mode-resolved scatter plot and per-band box plots,
and writes the underlying band statistics to CSV.  Frequencies close to Gamma
(the zero-frequency acoustic modes) are omitted to avoid a numerical division
by zero.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


DEFAULT_INPUT = Path(
    "ifc3_iter28_30_700conf/iteration_30/ifc3_rc3_4A/"
    "phono3py_wte_rta_qg_8x8x8_sigma_0.1THz/kappa-m888-s0.1.hdf5"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"phono3py kappa HDF5 (default: {DEFAULT_INPUT}).")
    parser.add_argument("--frequency-min", type=float, default=0.05, metavar="THZ", help="Exclude frequencies below this value (default: 0.05 THz).")
    parser.add_argument("--frequency-max", type=float, default=2.0, metavar="THZ", help="Only plot frequencies up to this value (default: 2.0 THz).")
    parser.add_argument("--temperature-index", type=int, default=0, help="Index along the HDF5 gamma temperature axis (default: 0).")
    parser.add_argument("--output", type=Path, help="Output PNG (default: <input directory>/gamma_over_omega_by_band.png).")
    parser.add_argument("--summary", type=Path, help="Output band-statistics CSV (default: next to the PNG).")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_lfs_pointer(path: Path) -> Path:
    """Use an identical local HDF5 when ``path`` is an unfetched Git-LFS pointer."""
    if not path.is_file():
        raise FileNotFoundError(f"Input HDF5 was not found: {path}")
    header = path.read_bytes()[:256]
    match = re.search(rb"oid sha256:([0-9a-f]{64})", header)
    if not match:
        return path

    expected_hash = match.group(1).decode()
    for candidate in Path.cwd().rglob(path.name):
        if candidate == path or candidate.stat().st_size < 1024:
            continue
        if sha256(candidate) == expected_hash:
            print(f"Input is a Git-LFS pointer; using identical local file: {candidate}")
            return candidate
    raise RuntimeError(
        f"{path} is a Git-LFS pointer, but its HDF5 content is unavailable locally. "
        "Fetch it with `git lfs pull` or supply a downloaded file using --input."
    )


def read_modes(path: Path, temperature_index: int) -> tuple[np.ndarray, np.ndarray, float | None, np.ndarray | None]:
    with h5py.File(path, "r") as data:
        if "frequency" not in data or "gamma" not in data:
            raise ValueError(f"{path} must contain both frequency and gamma datasets.")
        frequency = np.asarray(data["frequency"], dtype=float)
        gamma_values = np.asarray(data["gamma"], dtype=float)
        if gamma_values.ndim == 3:
            if not 0 <= temperature_index < gamma_values.shape[0]:
                raise IndexError(f"--temperature-index must be 0 to {gamma_values.shape[0] - 1}.")
            gamma = gamma_values[temperature_index]
        elif gamma_values.ndim == 2:
            gamma = gamma_values
        else:
            raise ValueError(f"Unexpected gamma shape {gamma_values.shape}; expected (temperature, grid, band).")
        if frequency.shape != gamma.shape or frequency.ndim != 2:
            raise ValueError(f"frequency shape {frequency.shape} and gamma shape {gamma.shape} are incompatible.")
        temperature = float(np.asarray(data["temperature"])[temperature_index]) if "temperature" in data else None
        mesh = np.asarray(data["mesh"], dtype=int) if "mesh" in data else None
    return frequency, gamma, temperature, mesh


def write_summary(path: Path, frequency: np.ndarray, ratio: np.ndarray, selected: np.ndarray) -> list[tuple[int, np.ndarray, np.ndarray]]:
    bands: list[tuple[int, np.ndarray, np.ndarray]] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("band", "mode_count", "frequency_min_THz", "frequency_median_THz", "frequency_max_THz", "gamma_over_omega_median", "gamma_over_omega_p95", "gamma_over_omega_max"))
        for index in range(frequency.shape[1]):
            mask = selected[:, index]
            if not np.any(mask):
                continue
            band_frequency = frequency[:, index][mask]
            band_ratio = ratio[:, index][mask]
            bands.append((index + 1, band_frequency, band_ratio))
            writer.writerow((
                index + 1,
                len(band_ratio),
                f"{band_frequency.min():.10g}", f"{np.median(band_frequency):.10g}", f"{band_frequency.max():.10g}",
                f"{np.median(band_ratio):.10g}", f"{np.percentile(band_ratio, 95):.10g}", f"{band_ratio.max():.10g}",
            ))
    return bands


def plot(bands: list[tuple[int, np.ndarray, np.ndarray]], output: Path, minimum: float, maximum: float, temperature: float | None, mesh: np.ndarray | None) -> None:
    band_numbers = np.asarray([band for band, _, _ in bands])
    colormap = plt.get_cmap("turbo")
    normalizer = Normalize(vmin=band_numbers.min(), vmax=band_numbers.max())
    figure, (scatter_axis, box_axis) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=(1.25, 1), constrained_layout=True)

    for band, frequencies, ratios in bands:
        scatter_axis.scatter(frequencies, ratios, s=16, alpha=0.75, color=colormap(normalizer(band)), edgecolors="none")
    colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=normalizer, cmap=colormap), ax=scatter_axis, pad=0.01)
    colorbar.set_label("phonon band index (1-based)")
    scatter_axis.set(xlabel="phonon frequency (THz)", ylabel=r"$\Gamma / \omega$", xlim=(minimum, maximum))
    scatter_axis.grid(alpha=0.3)

    box_axis.boxplot([ratios for _, _, ratios in bands], tick_labels=[str(band) for band, _, _ in bands], showfliers=False)
    box_axis.set(xlabel="phonon band index", ylabel=r"$\Gamma / \omega$ (box: Q1–Q3; line: median)")
    box_axis.grid(axis="y", alpha=0.3)

    details = [f"{minimum:g}–{maximum:g} THz"]
    if temperature is not None:
        details.append(f"T = {temperature:g} K")
    if mesh is not None:
        details.append("q = " + "x".join(map(str, mesh)))
    figure.suptitle("Low-frequency phonon linewidth ratio by band (" + ", ".join(details) + ")")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240)
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    if not 0.0 < arguments.frequency_min < arguments.frequency_max:
        raise ValueError("Require 0 < --frequency-min < --frequency-max.")
    source = resolve_lfs_pointer(arguments.input)
    frequency, gamma, temperature, mesh = read_modes(source, arguments.temperature_index)
    selected = np.isfinite(frequency) & np.isfinite(gamma) & (frequency >= arguments.frequency_min) & (frequency <= arguments.frequency_max)
    ratio = np.divide(gamma, frequency, out=np.full_like(gamma, np.nan), where=selected)
    output = arguments.output or arguments.input.with_name("gamma_over_omega_by_band.png")
    summary = arguments.summary or output.with_suffix(".csv")
    bands = write_summary(summary, frequency, ratio, selected)
    if not bands:
        raise ValueError("No modes are within the requested frequency window.")
    plot(bands, output, arguments.frequency_min, arguments.frequency_max, temperature, mesh)
    print(f"Wrote {output}\nWrote {summary}\nSelected bands: {', '.join(str(band) for band, _, _ in bands)}")


if __name__ == "__main__":
    main()
