#!/usr/bin/env python3
"""Plot iteration-resolved Wigner thermal conductivity from a summary CSV.

The input CSV is written by ``fit_kappa_with_wigner_iterations.py``.  Each
panel represents one q-point mesh and solver; the particle, coherence, and
total thermal conductivities are plotted against the TDEP iteration number.
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


DEFAULT_CSV = Path("ifc3_iter28_30_700conf/kappa_wigner_700conf_by_iteration.csv")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Input kappa summary CSV (default: {DEFAULT_CSV}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image path (default: <input directory>/kappa_by_iteration.png).",
    )
    parser.add_argument(
        "--solver",
        choices=("SMA", "exact"),
        nargs="+",
        help="Solvers to plot (default: all solvers in the CSV).",
    )
    parser.add_argument(
        "--q-label",
        nargs="+",
        help="Only plot these q-point mesh labels, e.g. --q-label 5x5x7 8x8x8.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV was not found: {path}")

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "iteration",
            "q_label",
            "solver",
            "status",
            "kappa_particle_iso",
            "kappa_coherence_iso",
            "kappa_total_iso",
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


def mesh_sort_key(label: str) -> tuple[int, int, int]:
    try:
        return tuple(int(value) for value in label.split("x"))  # type: ignore[return-value]
    except ValueError:
        return (999, 999, 999)


def plot(rows: list[dict[str, object]], output: Path) -> None:
    solver_order = [solver for solver in ("SMA", "exact") if any(row["solver"] == solver for row in rows)]
    solver_order.extend(sorted({str(row["solver"]) for row in rows}.difference(solver_order)))
    mesh_order = sorted({str(row["q_label"]) for row in rows}, key=mesh_sort_key)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["solver"]), str(row["q_label"])].append(row)

    figure, axes = plt.subplots(
        len(solver_order), len(mesh_order), squeeze=False, sharex=True, sharey=True,
        figsize=(4.2 * len(mesh_order), 3.5 * len(solver_order)),
    )
    series = (
        ("kappa_particle_iso", "particle", "#2878b5"),
        ("kappa_coherence_iso", "coherence", "#df7c35"),
        ("kappa_total_iso", "total", "#202020"),
    )
    for row_index, solver in enumerate(solver_order):
        for column_index, mesh in enumerate(mesh_order):
            axis = axes[row_index, column_index]
            data = sorted(grouped[solver, mesh], key=lambda row: int(row["iteration"]))
            for column, label, color in series:
                points = [(int(row["iteration"]), row[column]) for row in data if row[column] is not None]
                if points:
                    x_values, y_values = zip(*points)
                    axis.plot(x_values, y_values, "o-", color=color, linewidth=2, markersize=5, label=label)
            axis.set_title(f"{solver}, q = {mesh}")
            axis.grid(alpha=0.3)
            axis.set_xticks(sorted({int(row["iteration"]) for row in data}))
            if row_index == len(solver_order) - 1:
                axis.set_xlabel("TDEP iteration")
            if column_index == 0:
                axis.set_ylabel(r"$\kappa$ (W m$^{-1}$ K$^{-1}$)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(labels),
        frameon=False,
    )
    figure.suptitle("Wigner thermal conductivity by TDEP iteration", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.87))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    rows = read_rows(arguments.input)
    if arguments.solver:
        rows = [row for row in rows if row["solver"] in arguments.solver]
    if arguments.q_label:
        rows = [row for row in rows if row["q_label"] in arguments.q_label]
    if not rows:
        raise ValueError("No rows remain after applying --solver and --q-label filters.")

    output = arguments.output or arguments.input.with_name("kappa_by_iteration.png")
    plot(rows, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
