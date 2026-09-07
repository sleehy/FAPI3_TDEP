#!/usr/bin/env python3
"""Generate reviewed IFC3 samples from final TDEP IFC2, then fit IFC2+IFC3.

The final ``outfile.forceconstant`` from ``run_tdep.py`` supplies the sampling
Hamiltonian. New configurations are generated with TDEP's official
``canonical_configuration``, labelled and manually screened with the same
review functions used by ``run_tdep.py``, then fitted with TDEP's official
``extract_forceconstants --thirdorder_cutoff``. The original final iteration
is never changed.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path

import numpy as np
import yaml
from ase.io import read

# Reuse the established TDEP-native sampling, SevenNet labelling, interactive
# review, resampling, and data-file writing workflow. This avoids a divergent
# duplicate implementation for the IFC3 dataset.
from run_tdep import (
    canonical_configuration_command,
    generate_quantum_configurations,
    load_config,
    review_configurations,
    sevennet_calculator,
    tdep_executable,
    write_tdep_dataset,
)


FIT_INPUTS = (
    "infile.ucposcar",
    "infile.ssposcar",
    "infile.meta",
    "infile.stat",
    "infile.positions",
    "infile.forces",
)


def read_poscar_lattice(path: Path) -> np.ndarray:
    """Read a POSCAR lattice in Å, including VASP's scale-factor convention."""
    lines = path.read_text().splitlines()
    if len(lines) < 5:
        raise ValueError(f"{path} is too short to be a POSCAR.")
    try:
        scale = float(lines[1].split()[0])
        lattice = np.array([[float(value) for value in lines[row].split()[:3]] for row in range(2, 5)])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Could not parse the lattice from {path}.") from error
    volume = abs(float(np.linalg.det(lattice)))
    if scale == 0.0 or volume < 1.0e-12:
        raise ValueError(f"{path} has an invalid lattice.")
    if scale < 0.0:  # Negative POSCAR scale means target volume in Å³.
        scale = (-scale / volume) ** (1.0 / 3.0)
    return lattice * scale


def supercell_safe_cutoff(ssposcar: Path) -> float:
    """Return the largest non-aliased real-space cutoff for the supercell."""
    cell = read_poscar_lattice(ssposcar)
    volume = abs(float(np.linalg.det(cell)))
    heights = [volume / np.linalg.norm(np.cross(cell[(axis + 1) % 3], cell[(axis + 2) % 3])) for axis in range(3)]
    return 0.5 * min(heights)


def read_temperature(meta_file: Path) -> float:
    """Read the target temperature from a native TDEP ``infile.meta`` file."""
    values = [line.split() for line in meta_file.read_text().splitlines() if line.strip()]
    try:
        return float(values[3][0])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Could not read the temperature from {meta_file}.") from error


def cutoff_from_config(config_file: Path) -> float:
    """Read the exact IFC2 cutoff originally passed by ``run_tdep.py``."""
    try:
        with config_file.open() as handle:
            config = yaml.safe_load(handle)
        cutoff = float(config["tdep"]["secondorder_cutoff_A"])
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read tdep.secondorder_cutoff_A from {config_file}.") from error
    if cutoff <= 0.0:
        raise ValueError(f"tdep.secondorder_cutoff_A in {config_file} must be positive.")
    return cutoff


def validate_final_ifc2(source_dir: Path) -> Path:
    """Return the final IFC2 used to generate the new IFC3 ensemble."""
    for filename in ("outfile.forceconstant", "infile.forceconstant"):
        forceconstant = source_dir / filename
        if forceconstant.is_file():
            break
    else:
        raise FileNotFoundError(f"No final IFC2 was found in {source_dir}.")
    required = ("infile.ucposcar", "infile.ssposcar", "infile.meta")
    missing = [name for name in required if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{source_dir} is missing required file(s): {', '.join(missing)}")
    return forceconstant


def stage_sampling_inputs(source_dir: Path, final_ifc2: Path, work_dir: Path) -> None:
    """Link final IFC2 and structures into an isolated IFC3 work directory."""
    work_dir.mkdir(parents=True, exist_ok=True)
    links = {
        "infile.ucposcar": source_dir / "infile.ucposcar",
        "infile.ssposcar": source_dir / "infile.ssposcar",
        # canonical_configuration reads this exact file to sample the final
        # self-consistent IFC2 ensemble.
        "infile.forceconstant": final_ifc2,
    }
    for name, source in links.items():
        target = work_dir / name
        source = source.resolve()
        if target.is_symlink() and target.resolve() == source:
            continue
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to replace existing {target}.")
        target.symlink_to(source)


def validate_fitting_dataset(iteration_dir: Path) -> tuple[int, int]:
    """Check the reviewed data handed to ``extract_forceconstants``."""
    missing = [name for name in FIT_INPUTS if not (iteration_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{iteration_dir} is missing native TDEP fitting input(s): {', '.join(missing)}")
    meta = [line.split() for line in (iteration_dir / "infile.meta").read_text().splitlines() if line.strip()]
    try:
        n_atoms, n_configurations = int(meta[0][0]), int(meta[1][0])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Could not read atom/configuration counts from {iteration_dir / 'infile.meta'}.") from error
    expected_rows = n_atoms * n_configurations
    for filename in ("infile.positions", "infile.forces"):
        values = np.atleast_2d(np.loadtxt(iteration_dir / filename))
        if values.shape != (expected_rows, 3):
            raise ValueError(
                f"{iteration_dir / filename} has shape {values.shape}; expected ({expected_rows}, 3) from infile.meta."
            )
    return n_atoms, n_configurations


def configuration_paths(iteration_dir: Path) -> list[Path]:
    """Return numbered TDEP snapshot POSCARs in the fitting order."""
    snapshots = [
        path for path in iteration_dir.glob("contcar_conf*")
        if path.is_file() and path.name.removeprefix("contcar_conf").isdigit()
    ]
    return sorted(snapshots, key=lambda path: int(path.name.removeprefix("contcar_conf")))


def read_review_csv(
    path: Path,
    expected_configurations: list[Path],
    required_columns: set[str],
    *,
    ordered: bool,
) -> list[dict[str, str]]:
    """Read and cross-check one review table generated by ``run_tdep.py``."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing configuration-review file: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"{path} lacks required columns: {', '.join(sorted(required_columns))}")
        rows = list(reader)
    expected_names = [snapshot.name for snapshot in expected_configurations]
    names = [row.get("configuration", "") for row in rows]
    matches = names == expected_names if ordered else len(names) == len(expected_names) and set(names) == set(expected_names)
    if not matches:
        raise ValueError(f"{path} does not correspond exactly to the numbered snapshots in {path.parent}.")
    return rows


def validate_configuration_review(iteration_dir: Path, n_atoms: int, n_configurations: int) -> None:
    """Verify that reviewed snapshots and fitting data are identical under PBC."""
    snapshots = configuration_paths(iteration_dir)
    if len(snapshots) != n_configurations:
        raise ValueError(
            f"Found {len(snapshots)} numbered contcar_conf files, but infile.meta records {n_configurations} configurations."
        )
    positions = np.loadtxt(iteration_dir / "infile.positions").reshape(n_configurations, n_atoms, 3)
    reference_cell = np.asarray(read(iteration_dir / "infile.ssposcar", format="vasp").cell, dtype=float)
    maximum_position_error = 0.0
    for configuration_index, snapshot in enumerate(snapshots):
        atoms = read(snapshot, format="vasp")
        if len(atoms) != n_atoms or not np.allclose(atoms.cell, reference_cell, rtol=0.0, atol=1.0e-7):
            raise ValueError(f"{snapshot} does not match the reference supercell.")
        difference = atoms.get_scaled_positions(wrap=False) - positions[configuration_index]
        difference -= np.rint(difference)
        maximum_position_error = max(maximum_position_error, float(np.max(np.abs(difference))))
    if maximum_position_error > 2.0e-7:
        raise ValueError(
            "The reviewed snapshots do not match infile.positions "
            f"(maximum fractional-coordinate error {maximum_position_error:.3e})."
        )
    pb_rows = read_review_csv(
        iteration_dir / "pb_i_distance_screening.csv",
        snapshots,
        {"configuration", "short_pb_i_contacts", "long_reference_pb_i_bonds", "status"},
        ordered=True,
    )
    read_review_csv(
        iteration_dir / "mlp_energies.csv",
        snapshots,
        {"configuration", "mlp_energy_eV", "mlp_energy_eV_per_atom"},
        ordered=False,  # The energy review is deliberately sorted by energy.
    )
    flagged = [row["configuration"] for row in pb_rows if row.get("status", "").strip().lower() != "ok"]
    print(
        f"Configuration review preflight: {n_configurations} new snapshots match infile.positions; "
        f"maximum periodic fractional-coordinate error {maximum_position_error:.3e}."
    )
    if flagged:
        print(
            "WARNING: Pb-I geometry screening flagged "
            f"{len(flagged)} snapshot(s): {', '.join(flagged)}. "
            "This is warning-only, matching run_tdep.py's manual-review policy."
        )
    else:
        print("Configuration review preflight: Pb-I screen is clear and the MLP energy review covers every snapshot.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path, help="Final completed TDEP iteration containing outfile.forceconstant.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("tdep_tetragonal.yaml"),
        help="Active run_tdep.py YAML for new sampling and SevenNet labels (default: tdep_tetragonal.yaml).",
    )
    parser.add_argument("--thirdorder-cutoff", required=True, type=float, metavar="ANGSTROM", help="IFC3 cutoff in Å.")
    parser.add_argument(
        "--configurations",
        type=int,
        help="Number of newly sampled IFC3 configurations (default: sampling.configurations_per_iteration).",
    )
    parser.add_argument(
        "--secondorder-cutoff",
        type=float,
        metavar="ANGSTROM",
        help="Override the final run_tdep.py IFC2 cutoff. Normally this should not be set.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth reviewed configuration in the TDEP fit (default: 1).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="IFC3 sampling/fitting directory (default: <result-dir>/ifc3_rc3_<cutoff>A_nconf_<count>).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print TDEP commands without generating configurations or fitting.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.thirdorder_cutoff <= 0.0:
        raise ValueError("--thirdorder-cutoff must be positive.")
    if args.secondorder_cutoff is not None and args.secondorder_cutoff <= 0.0:
        raise ValueError("--secondorder-cutoff must be positive.")
    if args.configurations is not None and args.configurations < 1:
        raise ValueError("--configurations must be a positive integer.")
    if args.stride < 1:
        raise ValueError("--stride must be a positive integer.")

    source_dir = args.result_dir.resolve()
    final_ifc2 = validate_final_ifc2(source_dir)
    sampling_config_file = args.config.resolve()
    sampling_config = load_config(sampling_config_file)
    n_configurations = args.configurations or int(sampling_config["sampling"]["configurations_per_iteration"])
    # The shared run_tdep.py generator and resampler read this setting, so keep
    # the requested IFC3 ensemble size consistent through generation, review,
    # and any replacement snapshots.
    sampling_config["sampling"]["configurations_per_iteration"] = n_configurations
    temperature = read_temperature(source_dir / "infile.meta")
    configured_temperature = float(sampling_config["tdep"]["temperature_K"])
    if not np.isclose(configured_temperature, temperature, rtol=0.0, atol=1.0e-6):
        raise ValueError(
            f"{sampling_config_file} requests {configured_temperature:g} K, but final IFC2 was fitted at {temperature:g} K. "
            "Use the matching run_tdep.py config to sample the IFC3 ensemble."
        )

    source_config = source_dir.parent / "config_used.yaml"
    if args.secondorder_cutoff is not None:
        secondorder_cutoff = args.secondorder_cutoff
        cutoff_source = "--secondorder-cutoff"
    elif source_config.is_file():
        secondorder_cutoff = cutoff_from_config(source_config)
        cutoff_source = str(source_config)
    else:
        secondorder_cutoff = float(sampling_config["tdep"]["secondorder_cutoff_A"])
        cutoff_source = str(sampling_config_file)

    safe_cutoff = supercell_safe_cutoff(source_dir / "infile.ssposcar")
    for name, cutoff in (("secondorder", secondorder_cutoff), ("thirdorder", args.thirdorder_cutoff)):
        if cutoff > safe_cutoff + 1.0e-8:
            raise ValueError(f"{name} cutoff {cutoff:.3f} Å exceeds this supercell's safe maximum of {safe_cutoff:.3f} Å.")

    output_dir = (args.output_dir or source_dir / f"ifc3_rc3_{args.thirdorder_cutoff:g}A_nconf_{n_configurations}").resolve()
    if output_dir == source_dir:
        raise ValueError("--output-dir must differ from --result-dir so the final iteration is never overwritten.")
    extract_executable = tdep_executable(sampling_config, "extract_forceconstants")
    sampling_command = canonical_configuration_command(sampling_config, n_configurations, has_prior_fc=True)
    fit_command = [
        extract_executable,
        "--secondorder_cutoff", f"{secondorder_cutoff:.12g}",
        "--thirdorder_cutoff", f"{args.thirdorder_cutoff:.12g}",
        "--temperature", f"{temperature:.12g}",
        "--stride", str(args.stride),
    ]
    print(f"Final IFC2 sampling Hamiltonian: {final_ifc2}")
    print(f"New IFC3 configurations: {n_configurations} at {temperature:g} K")
    print(f"IFC2 cutoff: {secondorder_cutoff:g} Å ({cutoff_source}); IFC3 cutoff: {args.thirdorder_cutoff:g} Å")
    print(f"Supercell-safe cutoff: {safe_cutoff:.3f} Å")
    print("+", " ".join(sampling_command))
    print("+", " ".join(fit_command))
    if args.dry_run:
        return

    thirdorder_output = output_dir / "outfile.forceconstant_thirdorder"
    if thirdorder_output.is_file():
        print(f"IFC3 already exists; reusing {thirdorder_output}")
        return
    stage_sampling_inputs(source_dir, final_ifc2, output_dir)
    configurations = generate_quantum_configurations(sampling_config, output_dir, has_prior_fc=True)
    calculator = sevennet_calculator(sampling_config)
    iteration_name = source_dir.name.removeprefix("iteration_")
    iteration_number = int(iteration_name) if iteration_name.isdigit() else 0
    atoms_list, energies, forces = review_configurations(
        sampling_config, iteration_number, output_dir, configurations, calculator, has_prior_fc=True
    )
    write_tdep_dataset(output_dir, atoms_list, energies, forces, temperature)
    n_atoms, checked_configurations = validate_fitting_dataset(output_dir)
    if checked_configurations != n_configurations:
        raise RuntimeError(f"Reviewed {checked_configurations} configurations; expected {n_configurations}.")
    validate_configuration_review(output_dir, n_atoms, checked_configurations)
    subprocess.run(fit_command, cwd=output_dir, check=True)
    if not thirdorder_output.is_file():
        raise RuntimeError("TDEP completed without writing outfile.forceconstant_thirdorder.")
    print(f"Wrote {thirdorder_output}")
    print(f"TDEP also wrote the sequential IFC2 fit: {output_dir / 'outfile.forceconstant'}")


if __name__ == "__main__":
    main()
