#!/usr/bin/env python3
"""Plot Wigner thermal conductivity versus q-point grid from a summary CSV.

Each panel represents one solver and plots the particle, coherence, and total
thermal conductivities for every available TDEP iteration.  The input format
is the CSV written by ``fit_kappa_with_wigner_iterations.py``.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT_CSV = Path("ifc3_iter28_30_700conf/kappa_wigner_700conf_by_iteration.csv")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV, help=f"Input CSV (default: {DEFAULT_CSV}).")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image path (default: <input directory>/kappa_by_qgrid.png).",
    )
    parser.add_argument("--solver", choices=("SMA", "exact"), nargs="+", help="Solvers to plot (default: all).")
    parser.add_argument("--iteration", type=int, nargs="+", help="Iterations to plot (default: all).")
    return parser.parse_args()


def mesh_sort_key(label: str) -> tuple[int, int, int]:
    try:
        return tuple(int(value) for value in label.split("x"))  # type: ignore[return-value]
    except ValueError:
        return (999, 999, 999)


def read_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV was not found: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "iteration", "q_label", "solver", "status", "kappa_particle_iso", "kappa_coherence_iso", "kappa_total_iso"
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = required.difference(reader.fieldnames or [])
            raise ValueError(f"{path} is missing required column(s): {', '.join(sorted(missing))}")
        rows = []
        for row in reader:
            if row["status"].strip().lower() != "ok":
                continue
            try:
                row["iteration"] = int(row["iteration"])
                for key in ("kappa_particle_iso", "kappa_coherence_iso", "kappa_total_iso"):
                    row[key] = float(row[key]) if row[key] else None
            except ValueError as error:
                raise ValueError(f"Could not parse kappa data in CSV row: {row}") from error
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows with status=ok were found in {path}.")
    return rows


def plot(rows: list[dict[str, object]], output: Path) -> None:
    solvers = [solver for solver in ("SMA", "exact") if any(row["solver"] == solver for row in rows)]
    solvers.extend(sorted({str(row["solver"]) for row in rows}.difference(solvers)))
    meshes = sorted({str(row["q_label"]) for row in rows}, key=mesh_sort_key)
    iterations = sorted({int(row["iteration"]) for row in rows})
    mesh_index = {mesh: index for index, mesh in enumerate(meshes)}
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["solver"]), int(row["iteration"])].append(row)

    figure, axes = plt.subplots(1, len(solvers), squeeze=False, sharey=True, figsize=(6.2 * len(solvers), 4.8))
    component_styles = (
        ("kappa_particle_iso", "particle", "-"),
        ("kappa_coherence_iso", "coherence", "--"),
        ("kappa_total_iso", "total", ":"),
    )
    colors = plt.get_cmap("tab10").colors
    for axis, solver in zip(axes[0], solvers):
        for iteration_index, iteration in enumerate(iterations):
            data = {str(row["q_label"]): row for row in grouped[solver, iteration]}
            for column, component, linestyle in component_styles:
                points = [
                    (mesh_index[mesh], data[mesh][column])
                    for mesh in meshes
                    if mesh in data and data[mesh][column] is not None
                ]
                if points:
                    x_values, y_values = zip(*points)
                    axis.plot(
                        x_values,
                        y_values,
                        marker="o",
                        linestyle=linestyle,
                        linewidth=2,
                        color=colors[iteration_index],
                    )
        axis.set_title(solver)
        axis.set_xticks(range(len(meshes)), meshes)
        axis.set_xlabel("q-point grid")
        axis.grid(alpha=0.3)
    axes[0, 0].set_ylabel(r"$\kappa$ (W m$^{-1}$ K$^{-1}$)")

    iteration_handles = [
        Line2D([], [], color=colors[index], marker="o", linewidth=2, label=f"iter {iteration}")
        for index, iteration in enumerate(iterations)
    ]
    component_handles = [
        Line2D([], [], color="black", linestyle=linestyle, linewidth=2, label=component)
        for _, component, linestyle in component_styles
    ]
    figure.legend(
        iteration_handles,
        [handle.get_label() for handle in iteration_handles],
        title="TDEP iteration",
        loc="upper center",
        bbox_to_anchor=(0.34, 0.97),
        ncol=len(iteration_handles),
        frameon=False,
    )
    figure.legend(
        component_handles,
        [handle.get_label() for handle in component_handles],
        title="component",
        loc="upper center",
        bbox_to_anchor=(0.78, 0.97),
        ncol=len(component_handles),
        frameon=False,
    )
    figure.suptitle("Wigner thermal conductivity versus q-point grid", y=0.998)
    figure.tight_layout(rect=(0, 0, 1, 0.79))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    rows = read_rows(arguments.input)
    if arguments.solver:
        rows = [row for row in rows if row["solver"] in arguments.solver]
    if arguments.iteration:
        rows = [row for row in rows if row["iteration"] in arguments.iteration]
    if not rows:
        raise ValueError("No rows remain after applying --solver and --iteration filters.")
    output = arguments.output or arguments.input.with_name("kappa_by_qgrid.png")
    plot(rows, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
