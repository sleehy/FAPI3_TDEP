#!/usr/bin/env python3
"""Run TDEP with automatic geometry-aware configuration resampling.

This is the non-interactive counterpart to ``run_tdep.py``.  It retains the
same TDEP sampling, SevenNet labelling, IFC2 fitting, and phonon-property
workflow, but automatically replaces unsafe configurations before fitting.

Only robust high-energy MLP outliers are inspected for N-H/C-H and H-Pb/H-I
geometry. All automatic screening is disabled for the first two iterations.
From iteration 3 onward, Pb-I problems are always replaced and H geometry is
checked only for high-energy outliers.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase.io import read

from plot_bond_distances import find_x_h_bonds, measure_bonds
from run_tdep import (
    calculate_phonon_properties,
    config_path,
    config_snapshot_matches,
    fit_force_constants,
    generate_quantum_configurations,
    label_configurations,
    load_config,
    plot_overlay,
    read_vibrational_free_energy,
    relabel_replacements,
    resample_configurations,
    selected_for_dispersion_overlay,
    sevennet_calculator,
    stage_tdep_inputs,
    tdep_executable,
    validate_configurations,
    validate_cutoff,
    write_energy_review,
    write_free_energy_history,
    write_pb_i_distance_screening,
    write_tdep_dataset,
    write_tdep_unitcell,
)


@dataclass(frozen=True)
class ScreeningLimits:
    """Geometry and energy thresholds for one automatic-review pass."""

    outlier_mad_z: float
    nhch_min_ratio: float
    nhch_max_ratio: float
    minimum_h_pb: float
    minimum_h_i: float
    screening_start_iteration: int
    max_resample_rounds: int


def high_energy_outliers(energies: np.ndarray, atoms_per_configuration: int, threshold: float) -> tuple[set[int], np.ndarray]:
    """Return one-based robust upper-tail energy outliers and their z scores.

    The modified z score uses the median and MAD of eV/atom. Only the upper
    tail is screened: a high-energy snapshot is a useful proxy for a distorted
    geometry, whereas a low-energy fluctuation is not by itself defective.
    The conventional 3.5 cutoff is deliberately conservative.
    """
    values = np.asarray(energies, dtype=float) / atoms_per_configuration
    scores = np.zeros(len(values), dtype=float)
    if len(values) < 4:
        return set(), scores
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 1.0e-12:
        scores = 0.67448975 * (values - median) / mad
    else:
        q25, q75 = np.percentile(values, [25, 75])
        robust_scale = float((q75 - q25) / 1.3489795)
        if robust_scale <= 1.0e-12:
            return set(), scores
        scores = (values - median) / robust_scale
    return set(np.flatnonzero(scores > threshold).astype(int) + 1), scores


def h_geometry_reason(atoms, reference_bonds, reference_lengths, limits: ScreeningLimits) -> str:
    """Return a semicolon-separated failure reason, or an empty string."""
    measured = measure_bonds(atoms, reference_bonds)
    ratios = np.concatenate(
        [measured[label] / reference_lengths[label] for label in ("N-H", "C-H")]
    )
    reasons: list[str] = []
    if np.any(ratios < limits.nhch_min_ratio) or np.any(ratios > limits.nhch_max_ratio):
        reasons.append(
            f"N-H/C-H ratio {ratios.min():.3f}-{ratios.max():.3f} outside "
            f"{limits.nhch_min_ratio:.2f}-{limits.nhch_max_ratio:.2f}"
        )

    symbols = np.asarray(atoms.get_chemical_symbols())
    hydrogen = np.flatnonzero(symbols == "H")
    for element, minimum in (("Pb", limits.minimum_h_pb), ("I", limits.minimum_h_i)):
        heavy = np.flatnonzero(symbols == element)
        if not len(hydrogen) or not len(heavy):
            raise ValueError(f"Automatic H-{element} screening requires both H and {element} atoms.")
        closest = min(float(np.min(atoms.get_distances(int(index), heavy, mic=True))) for index in hydrogen)
        if closest < minimum:
            reasons.append(f"minimum H-{element} {closest:.3f} A < {minimum:.3f} A")
    return "; ".join(reasons)


def write_auto_screening(
    iteration_dir: Path,
    round_number: int,
    configuration_files: list[Path],
    energies: np.ndarray,
    scores: np.ndarray,
    energy_outliers: set[int],
    h_failures: dict[int, str],
    pb_i_failures: set[int],
    pb_i_checked: bool,
) -> Path:
    """Record why every configuration was accepted, skipped, or resampled."""
    atoms_per_configuration = len(read(configuration_files[0], format="vasp"))
    filename = iteration_dir / f"automatic_screening_round_{round_number:03d}.csv"
    with filename.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "configuration", "mlp_energy_eV_per_atom", "modified_z_score", "energy_outlier",
                "h_geometry_checked", "h_geometry_reason", "pb_i_checked", "pb_i_flagged", "action",
            ]
        )
        for number, configuration in enumerate(configuration_files, start=1):
            rejected = number in h_failures or number in pb_i_failures
            writer.writerow(
                [
                    configuration.name,
                    f"{energies[number - 1] / atoms_per_configuration:.12e}",
                    f"{scores[number - 1]:.6f}",
                    "yes" if number in energy_outliers else "no",
                    "yes" if number in energy_outliers else "no",
                    h_failures.get(number, ""),
                    "yes" if pb_i_checked else "no",
                    "yes" if number in pb_i_failures else "no",
                    "resample" if rejected else "accept",
                ]
            )
    return filename


def automatic_review(
    config: dict,
    iteration: int,
    iteration_dir: Path,
    configuration_files: list[Path],
    calculator,
    has_prior_fc: bool,
    limits: ScreeningLimits,
):
    """Label, screen, and replace snapshots until the automatic policy passes."""
    atoms_list, energies, forces = label_configurations(configuration_files, calculator)
    atoms_per_configuration = len(atoms_list[0])
    if iteration < limits.screening_start_iteration:
        validate_configurations(atoms_list)
        write_energy_review(iteration_dir, configuration_files, energies)
        audit = write_auto_screening(
            iteration_dir, 1, configuration_files, energies, np.zeros(len(energies)), set(), {}, set(), False,
        )
        print(
            f"  All automatic screening skipped for iteration {iteration} "
            f"(starts at iteration {limits.screening_start_iteration}).\n"
            f"  Audit: {audit}",
            flush=True,
        )
        return atoms_list, energies, forces

    reference = read(iteration_dir / "infile.ssposcar", format="vasp")
    reference_bonds = find_x_h_bonds(reference)
    reference_lengths = measure_bonds(reference, reference_bonds)
    if any(np.any(lengths <= 0.0) for lengths in reference_lengths.values()):
        raise ValueError("The relaxed reference contains a zero N-H or C-H bond length.")
    for round_number in range(1, limits.max_resample_rounds + 1):
        validate_configurations(atoms_list)
        write_energy_review(iteration_dir, configuration_files, energies)
        energy_outliers, scores = high_energy_outliers(energies, atoms_per_configuration, limits.outlier_mad_z)
        h_failures = {
            number: h_geometry_reason(atoms_list[number - 1], reference_bonds, reference_lengths, limits)
            for number in sorted(energy_outliers)
        }
        h_failures = {number: reason for number, reason in h_failures.items() if reason}

        pb_i_checked = True
        pb_i_failures = set(write_pb_i_distance_screening(config, iteration_dir, configuration_files))

        rejected = sorted(set(h_failures) | pb_i_failures)
        audit = write_auto_screening(
            iteration_dir, round_number, configuration_files, energies, scores, energy_outliers,
            h_failures, pb_i_failures, pb_i_checked,
        )
        print(
            f"  Automatic screen round {round_number}: {len(energy_outliers)} high-energy outlier(s), "
            f"{len(h_failures)} H-geometry failure(s), {len(pb_i_failures)} Pb-I failure(s).\n"
            f"  Audit: {audit}",
            flush=True,
        )
        if not rejected:
            return atoms_list, energies, forces
        if round_number == limits.max_resample_rounds:
            raise RuntimeError(
                f"Automatic review exceeded --max-resample-rounds={limits.max_resample_rounds}; "
                f"still rejected: {', '.join(map(str, rejected))}."
            )
        print("  Resampling configuration(s): " + ", ".join(map(str, rejected)), flush=True)
        replaced = resample_configurations(config, iteration_dir, configuration_files, rejected, has_prior_fc)
        relabel_replacements(configuration_files, replaced, calculator, atoms_list, energies, forces)

    raise AssertionError("Automatic-review loop ended unexpectedly.")


def run(config: dict, config_file: Path, dry_run: bool, limits: ScreeningLimits) -> None:
    output = config_path(config, config["output"]["directory"])
    unitcell = read(config_path(config, config["input"]["structure"]), format="vasp")
    multiplier = int(np.prod(config["tdep"]["supercell_matrix"]))
    print(f"Supercell: {len(unitcell) * multiplier} atoms; samples per iteration: {config['sampling']['configurations_per_iteration']}")
    for executable in ("generate_structure", "canonical_configuration", "extract_forceconstants", "phonon_dispersion_relations"):
        tdep_executable(config, executable)
    if dry_run:
        print("Dry run complete. TDEP and SevenNet inputs were validated; no calculations were performed.")
        return

    output.mkdir(parents=True, exist_ok=True)
    config_snapshot = output / "config_used.yaml"
    if config_snapshot.exists() and not config_snapshot_matches(config_snapshot, config_file):
        raise RuntimeError(
            f"{output} was created with a different configuration. Choose a new output.directory "
            "or archive the existing output before running."
        )
    if not config_snapshot.exists() or config_snapshot.read_text() != config_file.read_text():
        shutil.copy2(config_file, config_snapshot)
    structure_files = write_tdep_unitcell(config, output / "structure")
    validate_cutoff(config, structure_files[1])
    calculator = sevennet_calculator(config)
    selected_bands: list[tuple[int, np.ndarray]] = []
    free_energy_rows: list[tuple[int, float]] = []

    for iteration in range(1, int(config["tdep"]["iterations"]) + 1):
        iteration_dir = output / f"iteration_{iteration:02d}"
        previous_fc = output / f"iteration_{iteration - 1:02d}" / "outfile.forceconstant" if iteration > 1 else None
        if previous_fc is not None and not previous_fc.is_file():
            raise RuntimeError(f"Missing force constants from iteration {iteration - 1}: {previous_fc}")
        stage_tdep_inputs(structure_files, iteration_dir, previous_fc)
        forceconstant = iteration_dir / "outfile.forceconstant"
        if not forceconstant.exists():
            configurations = generate_quantum_configurations(config, iteration_dir, has_prior_fc=previous_fc is not None)
            atoms_list, energies, forces = automatic_review(
                config, iteration, iteration_dir, configurations, calculator, previous_fc is not None, limits
            )
            write_tdep_dataset(iteration_dir, atoms_list, energies, forces, float(config["tdep"]["temperature_K"]))
            fit_force_constants(config, iteration_dir)
        else:
            shutil.copy2(forceconstant, iteration_dir / "infile.forceconstant")

        dispersion_file = iteration_dir / "outfile.dispersion_relations"
        free_energy_file = iteration_dir / "outfile.free_energy"
        bands = np.loadtxt(dispersion_file) if dispersion_file.exists() and free_energy_file.exists() else calculate_phonon_properties(config, iteration_dir)
        np.save(iteration_dir / "dispersion_relations_THz.npy", bands)
        free_energy_rows.append((iteration, read_vibrational_free_energy(iteration_dir, float(config["tdep"]["temperature_K"]))))
        write_free_energy_history(free_energy_rows, float(config["tdep"]["temperature_K"]), output)
        if selected_for_dispersion_overlay(iteration, config):
            selected_bands.append((iteration, bands))
            plot_overlay(selected_bands, config, output / "phonon_dispersion_by_iteration.png")

    with (output / "run_summary.json").open("w") as handle:
        json.dump(
            {
                "completed_tdep_iterations": len(free_energy_rows),
                "dispersion_overlay_iterations": [iteration for iteration, _ in selected_bands],
                "supercell_atoms": len(unitcell) * multiplier,
                "configuration_screening": "automatic",
            },
            handle,
            indent=2,
        )
    print(f"Finished {len(free_energy_rows)} requested TDEP iterations. Results: {output}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=Path("tdep_tetragonal.yaml"), type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without TDEP or SevenNet calculations.")
    parser.add_argument("--outlier-mad-z", type=float, default=3.5, help="Upper-tail robust energy outlier threshold (default: 3.5).")
    parser.add_argument("--nhch-min-ratio", type=float, default=0.8)
    parser.add_argument("--nhch-max-ratio", type=float, default=1.3)
    parser.add_argument("--minimum-h-pb", type=float, default=2.2, metavar="ANGSTROM")
    parser.add_argument("--minimum-h-i", type=float, default=1.8, metavar="ANGSTROM")
    parser.add_argument(
        "--screening-start-iteration", "--pb-i-start-iteration",
        dest="screening_start_iteration", type=int, default=3, metavar="N",
        help="Start all automatic screening at iteration N (default: 3).",
    )
    parser.add_argument("--max-resample-rounds", type=int, default=10, metavar="N")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.outlier_mad_z <= 0.0 or not 0.0 < args.nhch_min_ratio < args.nhch_max_ratio:
        raise ValueError("Energy threshold and N-H/C-H ratios must be positive and ordered.")
    if args.minimum_h_pb <= 0.0 or args.minimum_h_i <= 0.0:
        raise ValueError("Minimum H-Pb and H-I distances must be positive.")
    if args.screening_start_iteration < 1 or args.max_resample_rounds < 1:
        raise ValueError("Screening start iteration and maximum resample rounds must be positive.")
    config_file = args.config.resolve()
    limits = ScreeningLimits(
        args.outlier_mad_z,
        args.nhch_min_ratio,
        args.nhch_max_ratio,
        args.minimum_h_pb,
        args.minimum_h_i,
        args.screening_start_iteration,
        args.max_resample_rounds,
    )
    run(load_config(config_file), config_file, args.dry_run, limits)


if __name__ == "__main__":
    main()
