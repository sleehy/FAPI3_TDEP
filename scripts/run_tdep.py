#!/usr/bin/env python3
"""Run stochastic TDEP with native TDEP programs and SevenNet force labels.

TDEP owns the lattice-dynamics steps: ``generate_structure`` builds the
supercell, ``canonical_configuration --quantum`` samples it,
``extract_forceconstants`` fits FC2, and ``phonon_dispersion_relations``
calculates each iteration's bands. Python only handles config, SevenNet labels,
TDEP file conversion, and the convergence-overlay plot.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ase.io import read, write
from sevenn.calculator import SevenNetCalculator


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


def run_command(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def tdep_executable(config: dict, name: str) -> str:
    executable = config_path(config, config["tdep"]["bin_dir"]) / name
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(
            f"TDEP executable not found or not executable: {executable}. "
            "Initialize the submodule and run: bash scripts/setup_tdep.sh"
        )
    return str(executable)


def write_tdep_unitcell(config: dict, structure_dir: Path) -> tuple[Path, Path]:
    """Write strict VASP5 files accepted by TDEP (without Selective dynamics)."""
    structure_dir.mkdir(parents=True, exist_ok=True)
    ucposcar = structure_dir / "infile.ucposcar"
    ssposcar = structure_dir / "infile.ssposcar"
    if not ssposcar.exists():
        atoms = read(config_path(config, config["input"]["structure"]), format="vasp")
        write(ucposcar, atoms, format="vasp", direct=True, vasp5=True, sort=False)
        matrix = [str(value) for value in config["tdep"]["supercell_matrix"]]
        run_command([tdep_executable(config, "generate_structure"), "--dimensions", *matrix], structure_dir)
        (structure_dir / "outfile.ssposcar").replace(ssposcar)
    return ucposcar, ssposcar


def stage_tdep_inputs(structure_files: tuple[Path, Path], iteration_dir: Path, prior_fc: Path | None) -> None:
    iteration_dir.mkdir(parents=True, exist_ok=True)
    for source in structure_files:
        target = iteration_dir / source.name
        if not target.exists():
            shutil.copy2(source, target)
    if prior_fc is not None:
        shutil.copy2(prior_fc, iteration_dir / "infile.forceconstant")


def generate_quantum_configurations(config: dict, iteration_dir: Path, has_prior_fc: bool) -> list[Path]:
    existing = sorted(iteration_dir.glob("contcar_conf*"))
    wanted = int(config["sampling"]["configurations_per_iteration"])
    if len(existing) == wanted:
        return existing
    if existing:
        raise RuntimeError(f"Found {len(existing)} partial configurations in {iteration_dir}; expected {wanted}.")
    command = [
        tdep_executable(config, "canonical_configuration"),
        "--temperature", str(config["tdep"]["temperature_K"]),
        "--nconf", str(wanted),
    ]
    if bool(config["sampling"].get("quantum", False)):
        command.append("--quantum")
    if not has_prior_fc:
        command += ["--maximum_frequency", str(config["tdep"]["initial_maximum_frequency_THz"])]
    run_command(command, iteration_dir)
    generated = sorted(iteration_dir.glob("contcar_conf*"))
    if len(generated) != wanted:
        raise RuntimeError(f"TDEP generated {len(generated)} configurations, expected {wanted}.")
    return generated


def sevennet_calculator(config: dict) -> SevenNetCalculator:
    checkpoint = config_path(config, config["calculation"]["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SevenNet checkpoint was not found: {checkpoint}")
    return SevenNetCalculator(str(checkpoint), device=config["calculation"].get("device", "auto"))


def label_configurations(configuration_files: list[Path], calculator: SevenNetCalculator):
    atoms_list, energies, forces = [], [], []
    total = len(configuration_files)
    for index, filename in enumerate(configuration_files, start=1):
        atoms = read(filename, format="vasp")
        atoms.calc = calculator
        energies.append(atoms.get_potential_energy())
        forces.append(atoms.get_forces())
        atoms_list.append(atoms)
        if index == 1 or index % 10 == 0 or index == total:
            print(f"  SevenNet: {index}/{total}", flush=True)
    return atoms_list, np.asarray(energies), np.asarray(forces)


def write_tdep_dataset(iteration_dir: Path, atoms_list, energies: np.ndarray, forces: np.ndarray, temperature: float) -> None:
    """Convert SevenNet-labelled POSCAR snapshots to native TDEP input files."""
    positions = np.concatenate([atoms.get_scaled_positions(wrap=False) for atoms in atoms_list])
    np.savetxt(iteration_dir / "infile.positions", positions, fmt="%.16e")
    np.savetxt(iteration_dir / "infile.forces", forces.reshape(-1, 3), fmt="%.16e")
    np.save(iteration_dir / "energies_eV.npy", energies)
    np.save(iteration_dir / "forces_eV_per_A.npy", forces)

    n_atoms, n_conf = len(atoms_list[0]), len(atoms_list)
    (iteration_dir / "infile.meta").write_text(f"{n_atoms}\n{n_conf}\n0.0\n{temperature:.8f}\n")
    # TDEP requires an integer configuration index in column 1; the remaining
    # 12 columns (time, energies, T/P, and six stress components) are reals.
    with (iteration_dir / "infile.stat").open("w") as handle:
        for index, energy in enumerate(energies, start=1):
            values = [0.0, energy, energy, 0.0, temperature, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            handle.write(f"{index:d} " + " ".join(f"{value:.16e}" for value in values) + "\n")


def fit_force_constants(config: dict, iteration_dir: Path) -> Path:
    command = [
        tdep_executable(config, "extract_forceconstants"),
        "--secondorder_cutoff", str(config["tdep"]["secondorder_cutoff_A"]),
        "--temperature", str(config["tdep"]["temperature_K"]),
    ]
    run_command(command, iteration_dir)
    fitted = iteration_dir / "outfile.forceconstant"
    if not fitted.is_file():
        raise RuntimeError("TDEP did not create outfile.forceconstant.")
    shutil.copy2(fitted, iteration_dir / "infile.forceconstant")
    return fitted


def write_tdep_path(config: dict, iteration_dir: Path) -> None:
    disp = config["dispersion"]
    points, labels = disp["qpoints"], disp["labels"]
    if len(points) != len(labels) or len(points) < 2:
        raise ValueError("dispersion.qpoints and dispersion.labels must have equal length of at least two.")
    lines = ["CUSTOM", str(int(disp["points_per_segment"])), str(len(points) - 1)]
    for begin, end, start_label, end_label in zip(points[:-1], points[1:], labels[:-1], labels[1:]):
        coordinates = " ".join(f"{value:.10f}" for value in [*begin, *end])
        # TDEP's text parser expects ASCII labels; keep the Unicode Γ only in plots.
        start_label = str(start_label).replace("Γ", "GM")
        end_label = str(end_label).replace("Γ", "GM")
        lines.append(f"{coordinates} {start_label} {end_label}")
    (iteration_dir / "infile.qpoints_dispersion").write_text("\n".join(lines) + "\n")


def calculate_dispersion(config: dict, iteration_dir: Path) -> np.ndarray:
    write_tdep_path(config, iteration_dir)
    run_command([
        tdep_executable(config, "phonon_dispersion_relations"),
        "--readpath", "--nq_on_path", str(config["dispersion"]["points_per_segment"]), "--unit", "thz",
    ], iteration_dir)
    data = np.loadtxt(iteration_dir / "outfile.dispersion_relations")
    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError("Unexpected TDEP dispersion data.")
    return data


def plot_overlay(all_bands: list[tuple[int, np.ndarray]], config: dict, filename: Path) -> None:
    labels = config["dispersion"]["labels"]
    per_segment = int(config["dispersion"]["points_per_segment"])
    reference_x = all_bands[-1][1][:, 0]
    indices = [min(index * per_segment, len(reference_x) - 1) for index in range(len(labels))]
    ticks = reference_x[indices]
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(all_bands)))
    for (iteration, data), color in zip(all_bands, colors):
        ax.plot(data[:, 0], data[:, 1:], color=color, alpha=0.78, lw=0.55)
        ax.plot([], [], color=color, lw=2, label=f"iteration {iteration}")
    for boundary in ticks:
        ax.axvline(boundary, color="0.75", lw=0.7)
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set(xlim=(reference_x[0], reference_x[-1]), xticks=ticks, xticklabels=labels, ylabel="Frequency (THz)")
    ax.set_title(f"Native TDEP phonon renormalization, {config['tdep']['temperature_K']:g} K")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.savefig(filename, dpi=220)
    plt.close(fig)


def write_convergence(rows: list[dict], filename: Path) -> None:
    with filename.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iteration", "band_rms_change_THz"])
        writer.writeheader()
        writer.writerows(rows)


def run(config: dict, config_file: Path, dry_run: bool) -> None:
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
    shutil.copy2(config_file, output / "config_used.yaml")
    structure_files = write_tdep_unitcell(config, output / "structure")
    calculator = sevennet_calculator(config)
    all_bands: list[tuple[int, np.ndarray]] = []
    rows: list[dict] = []
    previous_bands: np.ndarray | None = None
    converged = False

    for iteration in range(1, int(config["tdep"]["iterations"]) + 1):
        iteration_dir = output / f"iteration_{iteration:02d}"
        previous_fc = output / f"iteration_{iteration - 1:02d}" / "outfile.forceconstant" if iteration > 1 else None
        if previous_fc is not None and not previous_fc.is_file():
            raise RuntimeError(f"Missing force constants from iteration {iteration - 1}: {previous_fc}")
        stage_tdep_inputs(structure_files, iteration_dir, previous_fc)
        forceconstant = iteration_dir / "outfile.forceconstant"
        if not forceconstant.exists():
            configurations = generate_quantum_configurations(config, iteration_dir, has_prior_fc=previous_fc is not None)
            atoms_list, energies, forces = label_configurations(configurations, calculator)
            write_tdep_dataset(iteration_dir, atoms_list, energies, forces, float(config["tdep"]["temperature_K"]))
            fit_force_constants(config, iteration_dir)
        else:
            shutil.copy2(forceconstant, iteration_dir / "infile.forceconstant")

        dispersion_file = iteration_dir / "outfile.dispersion_relations"
        bands = np.loadtxt(dispersion_file) if dispersion_file.exists() else calculate_dispersion(config, iteration_dir)
        np.save(iteration_dir / "dispersion_relations_THz.npy", bands)
        all_bands.append((iteration, bands))
        plot_overlay(all_bands, config, output / "phonon_dispersion_by_iteration.png")
        if previous_bands is not None:
            if bands.shape != previous_bands.shape:
                raise RuntimeError("TDEP dispersion grids changed between iterations.")
            change = float(np.sqrt(np.mean((bands[:, 1:] - previous_bands[:, 1:]) ** 2)))
            rows.append({"iteration": iteration, "band_rms_change_THz": change})
            write_convergence(rows, output / "convergence.csv")
            print(json.dumps(rows[-1], indent=2), flush=True)
            if iteration >= int(config["convergence"]["min_iterations"]) and change <= float(config["convergence"]["max_band_rms_change_THz"]):
                converged = True
                break
        previous_bands = bands.copy()

    with (output / "run_summary.json").open("w") as handle:
        json.dump({"converged": converged, "completed_tdep_iterations": len(all_bands), "supercell_atoms": len(unitcell) * multiplier}, handle, indent=2)
    print(f"Finished. Converged: {converged}. Results: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="tdep_tetragonal.yaml", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and settings without TDEP or SevenNet calculations.")
    args = parser.parse_args()
    config_file = args.config.resolve()
    try:
        run(load_config(config_file), config_file, args.dry_run)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
