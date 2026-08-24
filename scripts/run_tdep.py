#!/usr/bin/env python3
"""Config-driven self-consistent harmonic (TDEP-style) phonon workflow.

The workflow uses Phonopy's quantum canonical displacement generator, SevenNet
for single-point energy/force evaluations, and symfc for symmetry-constrained
second-order force matching.  Iteration 0 is a finite-displacement harmonic
calculation; iterations 1..N are the finite-temperature TDEP updates.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ase import Atoms
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from sevenn.calculator import SevenNetCalculator


def load_config(path: Path) -> dict:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The configuration must be a YAML mapping.")
    config["_config_dir"] = path.parent.resolve()
    return config


def path_from_config(config: dict, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config["_config_dir"] / path


def phonopy_to_ase(cell) -> Atoms:
    return Atoms(
        symbols=cell.symbols,
        scaled_positions=cell.scaled_positions,
        cell=cell.cell,
        pbc=True,
    )


def make_phonopy(config: dict) -> Phonopy:
    input_cfg = config["input"]
    unitcell = read_vasp(path_from_config(config, input_cfg["structure"]))
    return Phonopy(
        unitcell,
        supercell_matrix=config["tdep"]["supercell_matrix"],
        primitive_matrix=input_cfg["primitive_matrix"],
        symprec=float(input_cfg["symprec"]),
        calculator="vasp",
        log_level=1,
    )


def get_calculator(config: dict) -> SevenNetCalculator:
    calc_cfg = config["calculation"]
    checkpoint = path_from_config(config, calc_cfg["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SevenNet checkpoint was not found: {checkpoint}")
    return SevenNetCalculator(str(checkpoint), device=calc_cfg.get("device", "auto"))


def evaluate_structures(cells, calculator: SevenNetCalculator, label: str):
    """Evaluate energies and forces while preserving Phonopy's atom order."""
    energies, forces = [], []
    total = len(cells)
    for index, cell in enumerate(cells, start=1):
        atoms = phonopy_to_ase(cell)
        atoms.calc = calculator
        energies.append(atoms.get_potential_energy())
        forces.append(atoms.get_forces())
        if index == 1 or index % 10 == 0 or index == total:
            print(f"  {label}: {index}/{total}", flush=True)
    return np.asarray(energies), np.asarray(forces)


def build_initial_force_constants(config: dict, phonon: Phonopy, calculator, output: Path):
    """Build FC2 by SevenNet finite displacements for the first sampler."""
    fc_path = output / "iteration_00" / "force_constants.npy"
    if fc_path.exists():
        phonon.force_constants = np.load(fc_path)
        return

    print("Generating finite-displacement force constants (iteration 0).")
    phonon.generate_displacements(distance=float(config["tdep"]["finite_displacement_A"]))
    displaced = [cell for cell in phonon.supercells_with_displacements if cell is not None]
    print(f"  SevenNet force calculations required: {len(displaced)}")
    energies, forces = evaluate_structures(displaced, calculator, "initial FC2")
    phonon.forces = forces
    phonon.produce_force_constants(
        fc_calculator=config["tdep"].get("force_constant_fitter", "symfc"),
        calculate_full_force_constants=True,
        show_drift=True,
    )
    iteration_dir = fc_path.parent
    iteration_dir.mkdir(parents=True, exist_ok=True)
    np.save(fc_path, phonon.force_constants)
    np.save(iteration_dir / "energies_eV.npy", energies)


def displaced_cells(phonon: Phonopy, displacements: np.ndarray):
    reference = phonopy_to_ase(phonon.supercell)
    cells = []
    for displacement in displacements:
        atoms = reference.copy()
        atoms.positions += displacement
        cells.append(atoms)
    return cells


def calculate_dispersion(phonon: Phonopy, config: dict):
    disp_cfg = config["dispersion"]
    points = np.asarray(disp_cfg["qpoints"], dtype=float)
    npoints = int(disp_cfg["points_per_segment"])
    paths = [np.linspace(begin, end, npoints) for begin, end in zip(points[:-1], points[1:])]
    phonon.run_band_structure(paths)
    return [np.asarray(frequency) for frequency in phonon.band_structure.frequencies]


def plot_overlay(all_bands, config: dict, filename: Path):
    """Overlay all completed iterations on one high-symmetry band plot."""
    points = np.asarray(config["dispersion"]["qpoints"], dtype=float)
    labels = config["dispersion"]["labels"]
    reciprocal = 2 * np.pi * np.linalg.inv(np.asarray(make_phonopy(config).unitcell.cell)).T
    segment_lengths = [np.linalg.norm((end - begin) @ reciprocal) for begin, end in zip(points[:-1], points[1:])]
    bounds = np.r_[0.0, np.cumsum(segment_lengths)]

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(all_bands)))
    for (iteration, bands), color in zip(all_bands, colors):
        offset = 0.0
        for data, length in zip(bands, segment_lengths):
            x = offset + np.linspace(0.0, length, len(data))
            ax.plot(x, data, color=color, alpha=0.78, lw=0.55)
            offset += length
        ax.plot([], [], color=color, lw=2, label=f"iteration {iteration}")
    for boundary in bounds:
        ax.axvline(boundary, color="0.75", lw=0.7)
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_xlim(bounds[0], bounds[-1])
    ax.set_xticks(bounds)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Frequency (THz)")
    ax.set_title(f"TDEP phonon renormalization, {config['tdep']['temperature_K']:g} K")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def force_rmse(force_constants: np.ndarray, displacements: np.ndarray, forces: np.ndarray) -> float:
    predicted = -np.einsum("ijab,sjb->sia", force_constants, displacements, optimize=True)
    return float(np.sqrt(np.mean((forces - predicted) ** 2)))


def write_convergence(rows: list[dict], filename: Path):
    fields = ["iteration", "force_rmse_eV_per_A", "fc_relative_change", "band_rms_change_THz"]
    with filename.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(config: dict, config_path: Path, dry_run: bool = False):
    output = path_from_config(config, config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "config_used.yaml")
    phonon = make_phonopy(config)
    n_atoms = len(phonon.supercell)
    n_configs = int(config["sampling"]["configurations_per_iteration"])
    print(f"Supercell: {n_atoms} atoms; samples per TDEP iteration: {n_configs}")
    if dry_run:
        print("Dry run complete. No SevenNet calculations were performed.")
        return

    calculator = get_calculator(config)
    build_initial_force_constants(config, phonon, calculator, output)
    all_bands = [(0, calculate_dispersion(phonon, config))]
    np.savez_compressed(output / "iteration_00" / "bands.npz", *all_bands[-1][1])
    plot_overlay(all_bands, config, output / "phonon_dispersion_by_iteration.png")

    previous_fc = phonon.force_constants.copy()
    previous_flat_bands = np.concatenate(all_bands[-1][1]).ravel()
    rows = []
    converged = False
    seed0 = int(config["sampling"]["random_seed"])
    for iteration in range(1, int(config["tdep"]["iterations"]) + 1):
        iteration_dir = output / f"iteration_{iteration:02d}"
        iteration_dir.mkdir(exist_ok=True)
        fc_file = iteration_dir / "force_constants.npy"
        if fc_file.exists():
            print(f"Resuming existing iteration {iteration}.")
            phonon.force_constants = np.load(fc_file)
            displacements = np.load(iteration_dir / "displacements_A.npy")
            forces = np.load(iteration_dir / "forces_eV_per_A.npy")
        else:
            phonon.init_random_displacements(
                dist_func=config["sampling"]["statistics"],
                cutoff_frequency=float(config["sampling"]["cutoff_frequency_THz"]),
                max_distance=float(config["sampling"]["max_displacement_A"]),
            )
            displacements = phonon.get_random_displacements_at_temperature(
                float(config["tdep"]["temperature_K"]),
                number_of_snapshots=n_configs,
                random_seed=seed0 + iteration,
            )
            cells = displaced_cells(phonon, displacements)
            energies, forces = evaluate_structures(cells, calculator, f"TDEP iteration {iteration}")
            np.save(iteration_dir / "displacements_A.npy", displacements)
            np.save(iteration_dir / "forces_eV_per_A.npy", forces)
            np.save(iteration_dir / "energies_eV.npy", energies)
            phonon.dataset = {"displacements": displacements, "forces": forces, "supercell_energies": energies}
            phonon.produce_force_constants(
                fc_calculator=config["tdep"].get("force_constant_fitter", "symfc"),
                calculate_full_force_constants=True,
                show_drift=True,
            )
            np.save(fc_file, phonon.force_constants)

        bands = calculate_dispersion(phonon, config)
        np.savez_compressed(iteration_dir / "bands.npz", *bands)
        flat_bands = np.concatenate(bands).ravel()
        fc_change = float(np.linalg.norm(phonon.force_constants - previous_fc) / np.linalg.norm(previous_fc))
        band_change = float(np.sqrt(np.mean((flat_bands - previous_flat_bands) ** 2)))
        row = {
            "iteration": iteration,
            "force_rmse_eV_per_A": force_rmse(phonon.force_constants, displacements, forces),
            "fc_relative_change": fc_change,
            "band_rms_change_THz": band_change,
        }
        rows.append(row)
        write_convergence(rows, output / "convergence.csv")
        all_bands.append((iteration, bands))
        plot_overlay(all_bands, config, output / "phonon_dispersion_by_iteration.png")
        print(json.dumps(row, indent=2), flush=True)

        enough_iterations = iteration >= int(config["convergence"]["min_iterations"])
        if enough_iterations and fc_change <= float(config["convergence"]["max_fc_relative_change"]) and band_change <= float(config["convergence"]["max_band_rms_change_THz"]):
            converged = True
            break
        previous_fc = phonon.force_constants.copy()
        previous_flat_bands = flat_bands.copy()

    with (output / "run_summary.json").open("w") as handle:
        json.dump({"converged": converged, "completed_tdep_iterations": len(rows), "supercell_atoms": n_atoms}, handle, indent=2)
    print(f"Finished. Converged: {converged}. Results: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="tdep_tetragonal.yaml", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate structure/settings without ML evaluations.")
    args = parser.parse_args()
    config_path = args.config.resolve()
    try:
        run(load_config(config_path), config_path, dry_run=args.dry_run)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
