#!/usr/bin/env python3
"""Create a combined phonon-band and element-resolved-PDOS plot from TDEP.

The lattice dynamics are calculated by TDEP's official
``phonon_dispersion_relations`` executable.  This script only reads TDEP's
self-describing HDF5 outputs and arranges them in one Matplotlib figure; it
does not convert a TDEP force-constant file to a different format.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _decode(value: object) -> str:
    """Convert HDF5 byte/string attributes to a normal Python string."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return " ".join(_decode(item) for item in value.flat)
    return str(value)


def _frequency_first(values: np.ndarray, n_frequency: int, dataset: str) -> np.ndarray:
    """Move the frequency axis to axis 0, independent of HDF5 array order."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{dataset} must be two-dimensional, got shape {values.shape}.")
    matching_axes = [axis for axis, size in enumerate(values.shape) if size == n_frequency]
    if len(matching_axes) != 1:
        raise ValueError(
            f"Cannot identify the frequency axis of {dataset}: shape {values.shape}, "
            f"frequency grid length {n_frequency}."
        )
    return np.moveaxis(values, matching_axes[0], 0)


def _element_label(unique_atom_label: str) -> str:
    """Map TDEP labels such as ``Pb_2`` to their chemical element, ``Pb``."""
    element, separator, site_index = unique_atom_label.rpartition("_")
    return element if separator and site_index.isdigit() else unique_atom_label


def _sum_pdos_by_element(unique_atom_labels: list[str], pdos: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Sum TDEP unique-atom-site PDOS channels into chemical-element channels."""
    element_channels: dict[str, np.ndarray] = {}
    for unique_atom_label, channel in zip(unique_atom_labels, pdos.T):
        element = _element_label(unique_atom_label)
        if element not in element_channels:
            element_channels[element] = np.zeros_like(channel)
        element_channels[element] += channel
    return list(element_channels), np.column_stack(list(element_channels.values()))


def read_tdep_outputs(
    result_dir: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    str,
    str,
]:
    """Read native TDEP HDF5 band and unique-atom-site-PDOS data."""
    band_file = result_dir / "outfile.dispersion_relations.hdf5"
    dos_file = result_dir / "outfile.phonon_dos.hdf5"
    missing = [str(path) for path in (band_file, dos_file) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing TDEP output: " + ", ".join(missing))

    with h5py.File(band_file, "r") as handle:
        q_values = np.asarray(handle["q_values"], dtype=float).reshape(-1)
        q_ticks = np.asarray(handle["q_ticks"], dtype=float).reshape(-1)
        frequencies = _frequency_first(handle["frequencies"], len(q_values), "band frequencies")
        labels = _decode(handle.attrs["q_tick_labels"]).split()
        frequency_unit = _decode(handle["frequencies"].attrs["unit"])

    if len(labels) != len(q_ticks):
        raise ValueError(
            f"TDEP wrote {len(q_ticks)} q ticks but {len(labels)} labels; cannot label the band path."
        )

    with h5py.File(dos_file, "r") as handle:
        frequency_axis = np.asarray(handle["frequencies"], dtype=float).reshape(-1)
        total_dos = np.asarray(handle["dos"], dtype=float).reshape(-1)
        pdos = _frequency_first(handle["dos_per_unique_atom"], len(frequency_axis), "unique-atom PDOS")
        unique_atom_labels = _decode(handle.attrs["unique_atom_labels"]).split()
        dos_unit = _decode(handle["dos"].attrs["unit"])

    if len(total_dos) != len(frequency_axis):
        raise ValueError("TDEP DOS and frequency arrays have incompatible lengths.")
    if pdos.shape[1] != len(unique_atom_labels):
        raise ValueError(
            f"TDEP wrote {pdos.shape[1]} projected DOS channels but {len(unique_atom_labels)} unique-atom labels."
        )
    return (
        q_values,
        q_ticks,
        labels,
        frequencies,
        frequency_axis,
        total_dos,
        pdos,
        unique_atom_labels,
        frequency_unit,
        dos_unit,
    )


def run_tdep(result_dir: Path, executable: Path, qmesh: list[int], path_points: int, dos_points: int) -> None:
    """Ask the official TDEP executable to produce both inputs for the plot."""
    required = ("infile.ucposcar", "infile.forceconstant", "infile.qpoints_dispersion")
    missing = [name for name in required if not (result_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{result_dir} is not a TDEP iteration directory; missing: {', '.join(missing)}")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"TDEP executable was not found or is not executable: {executable}")

    command = [
        str(executable),
        "--readpath",
        "--nq_on_path",
        str(path_points),
        "--unit",
        "thz",
        "--dos",
        "--qpoint_grid",
        *(str(point) for point in qmesh),
        "--dospoints",
        str(dos_points),
    ]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=result_dir, check=True)


def plot_tdep_phonons(
    q_values: np.ndarray,
    q_ticks: np.ndarray,
    q_labels: list[str],
    bands: np.ndarray,
    frequency_axis: np.ndarray,
    total_dos: np.ndarray,
    pdos: np.ndarray,
    unique_atom_labels: list[str],
    frequency_unit: str,
    dos_unit: str,
    output: Path,
    title: str | None,
) -> None:
    """Plot bands and element-resolved PDOS on a shared y-axis."""
    fig, (band_axis, dos_axis) = plt.subplots(
        1,
        2,
        figsize=(11, 6),
        sharey=True,
        gridspec_kw={"width_ratios": (3.5, 1.35), "wspace": 0.05},
        constrained_layout=True,
    )
    band_axis.plot(q_values, bands, color="0.08", lw=0.65)
    for tick in q_ticks:
        band_axis.axvline(tick, color="0.78", lw=0.7, zorder=0)
    band_axis.axhline(0.0, color="0.35", lw=0.8)
    band_axis.set(
        xlim=(q_values[0], q_values[-1]),
        xticks=q_ticks,
        xticklabels=[label.replace("GM", "Γ").replace("G", "Γ") for label in q_labels],
    )
    band_axis.set_ylabel(f"Frequency ({frequency_unit})")
    band_axis.set_xlabel("Wave vector")
    band_axis.set_title(title or "TDEP phonon dispersion")

    dos_axis.plot(total_dos, frequency_axis, color="0.1", lw=1.3, ls="--", label="Total")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    element_labels, element_pdos = _sum_pdos_by_element(unique_atom_labels, pdos)
    

    summed_pdos = element_pdos.sum(axis=1)

    max_error = np.max(np.abs(summed_pdos - total_dos))
    relative_error = np.linalg.norm(summed_pdos - total_dos) / np.linalg.norm(total_dos)

    print("Elements:", element_labels)
    print(f"PDOS sum max error: {max_error:.3e}")
    print(f"PDOS sum relative error: {relative_error:.3e}")


    for index, (label, channel) in enumerate(zip(element_labels, element_pdos.T)):
        dos_axis.plot(channel, frequency_axis, lw=1.5, color=colors[index % len(colors)], label=label)
    dos_axis.axhline(0.0, color="0.35", lw=0.8)
    dos_axis.set_xlabel(f"DOS ({dos_unit})")
    dos_axis.set_xlim(left=0.0)
    dos_axis.tick_params(axis="y", left=False, labelleft=False)
    dos_axis.legend(frameon=False, fontsize=9, loc="best")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path, help="TDEP iteration directory containing infile.forceconstant.")
    parser.add_argument("--output", type=Path, help="Output PNG (default: <result-dir>/phonon_band_pdos.png).")
    parser.add_argument(
        "--tdep-executable",
        type=Path,
        default=Path("external/tdep/bin/phonon_dispersion_relations"),
        help="Official TDEP phonon_dispersion_relations executable.",
    )
    parser.add_argument("--qmesh", type=int, nargs=3, default=(24, 24, 24), metavar=("N1", "N2", "N3"))
    parser.add_argument("--path-points", type=int, default=81, help="q-points per high-symmetry path segment.")
    parser.add_argument("--dos-points", type=int, default=600, help="Number of points on the DOS frequency grid.")
    parser.add_argument("--title", help="Optional figure title.")
    parser.add_argument("--reuse", action="store_true", help="Reuse existing HDF5 outputs instead of recalculating them with TDEP.")
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    if not args.reuse:
        run_tdep(result_dir, args.tdep_executable.resolve(), list(args.qmesh), args.path_points, args.dos_points)

    q_values, q_ticks, q_labels, bands, frequency_axis, total_dos, pdos, unique_atom_labels, frequency_unit, dos_unit = read_tdep_outputs(result_dir)
    output = (args.output or result_dir / "phonon_band_pdos.png").resolve()
    plot_tdep_phonons(
        q_values,
        q_ticks,
        q_labels,
        bands,
        frequency_axis,
        total_dos,
        pdos,
        unique_atom_labels,
        frequency_unit,
        dos_unit,
        output,
        args.title,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
