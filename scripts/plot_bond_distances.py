#!/usr/bin/env python3
"""Plot covalent N-H and C-H bond-length distributions from VASP structures.

The bonds are selected in a reference POSCAR using covalent-radius cutoffs, then
the same atom-index pairs are measured in the target structure.  Keeping the
reference topology fixed makes the diagnostic robust to thermal displacements
and avoids accidentally reporting a neighbouring molecule as a new bond.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Sequence
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.data import atomic_numbers, covalent_radii
from ase.io import read


BondPair = tuple[int, int]
BondGroups = dict[str, list[BondPair]]


def default_cutoff(symbol: str, cutoff_scale: float) -> float:
    """Return an element-specific X-H covalent-bond cutoff in Å."""
    return cutoff_scale * (covalent_radii[atomic_numbers[symbol]] + covalent_radii[atomic_numbers["H"]])


def find_x_h_bonds(reference_atoms, cutoff_scale: float = 1.25) -> BondGroups:
    """Find C-H and N-H pairs in a reference structure under PBC.

    The reference must have the same atom order as every structure whose bond
    lengths will be measured.  A hydrogen is allowed to belong to only one
    selected pair; an ambiguous reference topology is rejected instead of
    silently double-counting a hydrogen.
    """
    if cutoff_scale <= 0:
        raise ValueError("cutoff_scale must be positive.")
    symbols = np.asarray(reference_atoms.get_chemical_symbols())
    hydrogen_indices = np.flatnonzero(symbols == "H")
    if not len(hydrogen_indices):
        raise ValueError("The reference structure contains no H atoms.")

    bonds: BondGroups = {"N-H": [], "C-H": []}
    assigned_hydrogens: dict[int, tuple[str, int]] = {}
    for element, label in (("N", "N-H"), ("C", "C-H")):
        cutoff = default_cutoff(element, cutoff_scale)
        for heavy_index in np.flatnonzero(symbols == element):
            distances = reference_atoms.get_distances(heavy_index, hydrogen_indices, mic=True)
            for hydrogen_index, distance in zip(hydrogen_indices, distances):
                if distance <= cutoff:
                    previous = assigned_hydrogens.get(int(hydrogen_index))
                    if previous is not None:
                        previous_label, previous_heavy = previous
                        raise ValueError(
                            "Ambiguous H bonding in the reference structure: "
                            f"H atom {hydrogen_index + 1} is within the covalent cutoff of "
                            f"both {previous_label} atom {previous_heavy + 1} and {label} atom {heavy_index + 1}. "
                            "Use a reference structure with unambiguous X-H bonds or lower --cutoff-scale."
                        )
                    bonds[label].append((int(heavy_index), int(hydrogen_index)))
                    assigned_hydrogens[int(hydrogen_index)] = (label, int(heavy_index))

    missing = {label: len(pairs) for label, pairs in bonds.items() if not pairs}
    if missing:
        present = ", ".join(f"{label}={count}" for label, count in bonds.items())
        raise ValueError(
            "Could not identify both N-H and C-H bonds in the reference structure "
            f"(found {present}). Increase --cutoff-scale only if these are genuinely covalent bonds."
        )
    return bonds


def validate_target_matches_reference(reference_atoms, target_atoms) -> None:
    if len(target_atoms) != len(reference_atoms):
        raise ValueError(
            f"Target has {len(target_atoms)} atoms but reference has {len(reference_atoms)}. "
            "They must have identical atom ordering."
        )
    if target_atoms.get_chemical_symbols() != reference_atoms.get_chemical_symbols():
        raise ValueError("Target and reference have different species or atom ordering.")


def measure_bonds(target_atoms, bonds: BondGroups) -> dict[str, np.ndarray]:
    """Measure fixed reference bond pairs in the target using minimum-image distances."""
    return {
        label: np.asarray(
            [target_atoms.get_distance(heavy_index, hydrogen_index, mic=True) for heavy_index, hydrogen_index in pairs],
            dtype=float,
        )
        for label, pairs in bonds.items()
    }


def write_bond_distance_histogram(
    reference: Path,
    structure: Path,
    output: Path | None = None,
    bins: int = 30,
    cutoff_scale: float = 1.25,
) -> tuple[Path, Path, dict[str, np.ndarray]]:
    """Write a two-panel N-H/C-H histogram and a CSV of its measured bonds."""
    if bins < 1:
        raise ValueError("bins must be a positive integer.")
    reference_atoms = read(reference, format="vasp")
    target_atoms = read(structure, format="vasp")
    validate_target_matches_reference(reference_atoms, target_atoms)
    bonds = find_x_h_bonds(reference_atoms, cutoff_scale=cutoff_scale)
    distances = measure_bonds(target_atoms, bonds)

    if output is None:
        output = structure.with_name(f"{structure.name}_bond_distance_histogram.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bond_type", "heavy_atom_index_1based", "hydrogen_atom_index_1based", "distance_A"])
        for label in ("N-H", "C-H"):
            for (heavy_index, hydrogen_index), distance in zip(bonds[label], distances[label]):
                writer.writerow([label, heavy_index + 1, hydrogen_index + 1, f"{distance:.10f}"])

    all_distances = np.concatenate([distances["N-H"], distances["C-H"]])
    lower = max(0.0, float(all_distances.min()) - 0.05)
    upper = float(all_distances.max()) + 0.05
    # Equal axes make N-H and C-H broadening directly comparable.
    edges = np.linspace(lower, upper, bins + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharex=True, constrained_layout=True)
    colors = {"N-H": "C1", "C-H": "C0"}
    for ax, label in zip(axes, ("N-H", "C-H")):
        values = distances[label]
        ax.hist(values, bins=edges, color=colors[label], edgecolor="white", linewidth=0.7)
        ax.axvline(values.mean(), color="0.2", linewidth=1.0, linestyle="--", label=f"mean {values.mean():.3f} Å")
        ax.set(title=f"{label} ({len(values)} bonds)", xlabel="Bond distance (Å)", ylabel="Count")
        ax.grid(axis="y", color="0.88", linewidth=0.7)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Covalent bond distances: {structure.name}")
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output, csv_path, distances


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", required=True, type=Path, help="Target POSCAR/CONTCAR whose bond lengths are plotted.")
    parser.add_argument(
        "--reference",
        type=Path,
        help="Reference POSCAR used to identify fixed C-H/N-H bonds. Defaults to --structure.",
    )
    parser.add_argument("--output", type=Path, help="Histogram PNG path (default: next to --structure).")
    parser.add_argument("--bins", type=int, default=30, help="Histogram bins (default: 30).")
    parser.add_argument(
        "--cutoff-scale",
        type=float,
        default=1.25,
        help="Covalent-radius multiplier for finding reference bonds (default: 1.25).",
    )
    args = parser.parse_args(argv)
    reference = args.reference if args.reference is not None else args.structure
    histogram, csv_path, distances = write_bond_distance_histogram(
        reference=reference,
        structure=args.structure,
        output=args.output,
        bins=args.bins,
        cutoff_scale=args.cutoff_scale,
    )
    print(f"Histogram: {histogram}")
    print(f"Bond distances: {csv_path}")
    print(
        "  ".join(
            f"{label}: n={len(values)}, mean={values.mean():.4f} Å, range={values.min():.4f}–{values.max():.4f} Å"
            for label, values in distances.items()
        )
    )


if __name__ == "__main__":
    main()
