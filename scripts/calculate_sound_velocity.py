#!/usr/bin/env python3
"""Fit acoustic-mode sound velocities from a completed TDEP band calculation.

Sound velocity is the long-wavelength limit ``v = dω/dk``.  This program
therefore fits the first few q-points of every path segment starting at Γ,
using the physical reciprocal-space distance reconstructed from
``infile.qpoints_dispersion`` and the primitive-cell lattice.  The first
three branches are reported as TA1, TA2, and LA in increasing fitted slope.

The TDEP text dispersion file is sufficient; TDEP is not rerun.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# f [THz] = (v / 2π) k, where k is in rad m⁻¹.  Thus a slope in
# THz Å equals 2π × 10² m s⁻¹.
THZ_PER_ANGSTROM_TO_M_PER_S = 2.0 * np.pi * 100.0
GAMMA_TOLERANCE = 1.0e-10


@dataclass(frozen=True)
class PathSegment:
    """One labelled segment from TDEP's CUSTOM q-point path."""

    start: np.ndarray
    end: np.ndarray
    start_label: str
    end_label: str

    @property
    def label(self) -> str:
        return f"{self.start_label}-{self.end_label}"


@dataclass(frozen=True)
class FitResult:
    """A zero-intercept acoustic dispersion fit."""

    slope_thz_per_angstrom: float
    slope_stderr_thz_per_angstrom: float
    r_squared: float

    @property
    def velocity_m_per_s(self) -> float:
        return self.slope_thz_per_angstrom * THZ_PER_ANGSTROM_TO_M_PER_S

    @property
    def velocity_stderr_m_per_s(self) -> float:
        return self.slope_stderr_thz_per_angstrom * THZ_PER_ANGSTROM_TO_M_PER_S


def _normalise_label(label: str) -> str:
    """Use TDEP's ASCII name for Gamma while accepting a Unicode input."""
    return label.strip().upper().replace("Γ", "GM").replace("GAMMA", "GM")


def read_tdep_path(path: Path) -> tuple[int, list[PathSegment]]:
    """Read a TDEP ``CUSTOM`` q-path file."""
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 4 or lines[0][0].upper() != "CUSTOM":
        raise ValueError(f"{path} is not a TDEP CUSTOM q-point path.")
    try:
        points_per_segment = int(lines[1][0])
        n_segments = int(lines[2][0])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Could not read the path dimensions in {path}.") from error
    if points_per_segment < 2 or len(lines) != n_segments + 3:
        raise ValueError(f"Unexpected number of segments in {path}.")

    segments = []
    for fields in lines[3:]:
        if len(fields) < 8:
            raise ValueError(f"Malformed q-path line in {path}: {' '.join(fields)}")
        try:
            start = np.array(fields[:3], dtype=float)
            end = np.array(fields[3:6], dtype=float)
        except ValueError as error:
            raise ValueError(f"Malformed q coordinates in {path}: {' '.join(fields)}") from error
        segments.append(PathSegment(start, end, _normalise_label(fields[6]), _normalise_label(fields[7])))
    return points_per_segment, segments


def read_poscar_lattice(path: Path) -> np.ndarray:
    """Read the 3-by-3 Cartesian lattice in Å from a VASP POSCAR/CONTCAR."""
    lines = path.read_text().splitlines()
    if len(lines) < 5:
        raise ValueError(f"{path} is too short to be a POSCAR.")
    try:
        scale = float(lines[1].split()[0])
        lattice = np.array([[float(value) for value in lines[index].split()[:3]] for index in range(2, 5)])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Could not read lattice vectors from {path}.") from error
    if lattice.shape != (3, 3) or abs(np.linalg.det(lattice)) < 1.0e-12:
        raise ValueError(f"{path} contains a singular lattice.")
    if scale == 0.0:
        raise ValueError(f"{path} has an invalid POSCAR scale factor of zero.")
    # A negative POSCAR scale specifies the desired cell volume in Å³.
    if scale < 0.0:
        scale = (-scale / abs(np.linalg.det(lattice))) ** (1.0 / 3.0)
    return lattice * scale


def reciprocal_lattice(lattice: np.ndarray) -> np.ndarray:
    """Return reciprocal vectors (row convention) in rad Å⁻¹."""
    return 2.0 * np.pi * np.linalg.inv(lattice).T


def fit_through_gamma(q_magnitude: np.ndarray, frequencies: np.ndarray) -> FitResult:
    """Fit f(k)=ak through Γ and estimate the standard error of a."""
    if len(q_magnitude) != len(frequencies) or len(q_magnitude) < 3:
        raise ValueError("At least three q-points, including Γ, are required for a sound-velocity fit.")
    if not np.isclose(q_magnitude[0], 0.0, atol=GAMMA_TOLERANCE):
        raise ValueError("The selected segment does not start at Γ.")
    denominator = float(np.dot(q_magnitude, q_magnitude))
    if denominator == 0.0:
        raise ValueError("All selected q-points coincide at Γ.")
    slope = float(np.dot(q_magnitude, frequencies) / denominator)
    fitted = slope * q_magnitude
    residual = frequencies - fitted
    residual_sum_squares = float(np.dot(residual, residual))
    # One fitted parameter (the slope); the intercept is fixed at the acoustic
    # sum-rule value f(Γ)=0.
    slope_stderr = float(np.sqrt(residual_sum_squares / (len(q_magnitude) - 1) / denominator))
    total_sum_squares = float(np.dot(frequencies, frequencies))
    r_squared = 1.0 - residual_sum_squares / total_sum_squares if total_sum_squares else 1.0
    return FitResult(slope, slope_stderr, r_squared)


def selected_gamma_segments(segments: list[PathSegment], requested: list[str] | None) -> list[tuple[int, PathSegment]]:
    """Select Γ-originating segments, optionally filtered by labels such as GM-X."""
    requested_labels = {_normalise_label(label) for label in requested} if requested else None
    selected = []
    for index, segment in enumerate(segments):
        is_gamma_start = np.allclose(segment.start, 0.0, rtol=0.0, atol=GAMMA_TOLERANCE)
        if not is_gamma_start:
            continue
        if requested_labels is not None and segment.label not in requested_labels:
            continue
        selected.append((index, segment))
    if requested_labels is not None:
        found = {segment.label for _, segment in selected}
        missing = sorted(requested_labels - found)
        if missing:
            raise ValueError(f"No Γ-originating path segment matches: {', '.join(missing)}")
    if not selected:
        raise ValueError("The q-point path contains no segment starting at Γ.")
    return selected


def plot_fits(rows: list[dict[str, object]], output: Path) -> None:
    """Plot each acoustic branch and its Γ-constrained linear fit."""
    directions = list(dict.fromkeys(str(row["direction"]) for row in rows))
    fig, axes = plt.subplots(1, len(directions), figsize=(5.4 * len(directions), 4.4), squeeze=False, constrained_layout=True)
    colors = {"TA1": "C0", "TA2": "C1", "LA": "C3"}
    for axis, direction in zip(axes[0], directions):
        direction_rows = [row for row in rows if row["direction"] == direction]
        for row in direction_rows:
            q = np.asarray(row["q_magnitude"], dtype=float)
            frequency = np.asarray(row["frequency"], dtype=float)
            branch = str(row["acoustic_mode"])
            color = colors[branch]
            axis.plot(q, frequency, "o", ms=4.5, color=color, label=f"{branch}: {row['velocity_m_per_s']:.0f} m/s")
            axis.plot(q, float(row["slope_thz_per_angstrom"]) * q, color=color, lw=1.2)
        axis.set(title=f"Γ → {direction.split('-', 1)[1]}", xlabel=r"$|q|$ (rad Å$^{-1}$)", ylabel="Frequency (THz)")
        axis.legend(frameon=False, fontsize=8)
        axis.grid(alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path, help="Completed TDEP iteration directory.")
    parser.add_argument(
        "--fit-points",
        type=int,
        default=8,
        help="Number of q-points from Γ used in each fit, including Γ (default: 8).",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        help="Γ-originating path labels to fit, e.g. GM-X GM-Z (default: all such segments).",
    )
    parser.add_argument("--output", type=Path, help="Output CSV (default: <result-dir>/sound_velocities.csv).")
    parser.add_argument("--plot", type=Path, help="Output PNG (default: <result-dir>/sound_velocities.png).")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.fit_points < 3:
        raise ValueError("--fit-points must be at least 3.")
    result_dir = args.result_dir.resolve()
    path_file = result_dir / "infile.qpoints_dispersion"
    structure_file = result_dir / "infile.ucposcar"
    dispersion_file = result_dir / "outfile.dispersion_relations"
    missing = [str(path) for path in (path_file, structure_file, dispersion_file) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing TDEP input/output: " + ", ".join(missing))

    points_per_segment, segments = read_tdep_path(path_file)
    if args.fit_points > points_per_segment:
        raise ValueError(f"--fit-points ({args.fit_points}) exceeds the {points_per_segment} q-points per segment.")
    bands = np.atleast_2d(np.loadtxt(dispersion_file))
    expected_rows = points_per_segment * len(segments)
    if bands.shape[0] != expected_rows or bands.shape[1] < 4:
        raise ValueError(
            f"Unexpected dispersion shape {bands.shape}; expected {expected_rows} rows and at least three phonon branches."
        )

    reciprocal = reciprocal_lattice(read_poscar_lattice(structure_file))
    rows: list[dict[str, object]] = []
    for segment_index, segment in selected_gamma_segments(segments, args.directions):
        fractional_q = np.linspace(segment.start, segment.end, points_per_segment)[: args.fit_points]
        q_magnitude = np.linalg.norm((fractional_q - segment.start) @ reciprocal, axis=1)
        frequencies = bands[segment_index * points_per_segment : (segment_index + 1) * points_per_segment, 1:4][
            : args.fit_points
        ]
        fits = [fit_through_gamma(q_magnitude, frequencies[:, branch]) for branch in range(3)]
        # TDEP orders frequencies at each q-point.  Naming by fitted velocity
        # yields the conventional TA1 <= TA2 <= LA ordering independently of
        # numerical swapping of degenerate transverse modes.
        for mode_name, branch, fit in zip(("TA1", "TA2", "LA"), np.argsort([fit.slope_thz_per_angstrom for fit in fits]), sorted(fits, key=lambda value: value.slope_thz_per_angstrom)):
            rows.append(
                {
                    "direction": segment.label,
                    "acoustic_mode": mode_name,
                    "branch_index": int(branch) + 1,
                    "fit_points": args.fit_points,
                    "q_max_rad_per_angstrom": float(q_magnitude[-1]),
                    "slope_thz_per_angstrom": fit.slope_thz_per_angstrom,
                    "slope_stderr_thz_per_angstrom": fit.slope_stderr_thz_per_angstrom,
                    "velocity_m_per_s": fit.velocity_m_per_s,
                    "velocity_stderr_m_per_s": fit.velocity_stderr_m_per_s,
                    "r_squared": fit.r_squared,
                    "q_magnitude": q_magnitude,
                    "frequency": frequencies[:, branch],
                }
            )

    output = (args.output or result_dir / "sound_velocities.csv").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "direction", "acoustic_mode", "branch_index", "fit_points", "q_max_rad_per_angstrom",
        "slope_thz_per_angstrom", "slope_stderr_thz_per_angstrom", "velocity_m_per_s",
        "velocity_stderr_m_per_s", "r_squared",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)

    plot = (args.plot or result_dir / "sound_velocities.png").resolve()
    plot_fits(rows, plot)
    print(f"Wrote {output}")
    print(f"Wrote {plot}")
    for row in rows:
        print(
            f"{row['direction']:>5s} {row['acoustic_mode']}: "
            f"{row['velocity_m_per_s']:.1f} ± {row['velocity_stderr_m_per_s']:.1f} m/s "
            f"(R²={row['r_squared']:.6f})"
        )


if __name__ == "__main__":
    main()
