#!/usr/bin/env python3
"""Fit iteration-resolved TDEP IFC2/IFC3 pairs and calculate Wigner kappa.

By default each selected iteration must contain TDEP's native labelled fitting
dataset. With ``--new-configurations``, a fresh labelled dataset is generated
from its final IFC2 instead. The script fits (or reuses) the joint IFC2/IFC3
pair, converts that exact pair to phono3py once, and then evaluates the
phono3py-WTE solver on the requested q mesh. Thus changing WTE settings never
refits the force constants.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import yaml

from tdep_to_phono3py import build_phono3py, convert_tdep_force_constants


FIT_INPUTS = (
    "infile.ucposcar",
    "infile.ssposcar",
    "infile.meta",
    "infile.stat",
    "infile.positions",
    "infile.forces",
)


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("ifc3_iter12_16_200conf"))
    parser.add_argument("--iterations", type=int, nargs="+", default=[12, 13, 14, 15, 16])
    configuration_source = parser.add_mutually_exclusive_group()
    configuration_source.add_argument(
        "--reuse-configurations",
        action="store_true",
        help="Fit the labelled native dataset already in each iteration (the default).",
    )
    configuration_source.add_argument(
        "--new-configurations",
        type=int,
        metavar="COUNT",
        help="Generate, label, and review COUNT new configurations from each iteration's final IFC2 before fitting.",
    )
    parser.add_argument(
        "--sampling-config",
        type=Path,
        default=Path("tdep_tetragonal.yaml"),
        help="run_tdep.py YAML for --new-configurations (default: tdep_tetragonal.yaml).",
    )
    parser.add_argument("--thirdorder-cutoff", required=True, type=float, metavar="ANGSTROM")
    parser.add_argument("--qpoint-grid", required=True, type=int, nargs=3, metavar=("N1", "N2", "N3"))
    parser.add_argument(
        "--supercell-matrix", type=int, nargs=3, metavar=("N1", "N2", "N3"),
        help="Default: read the diagonal matrix from <input-root>/config_used.yaml.",
    )
    parser.add_argument("--solver", choices=("rta", "lbte"), default="rta")
    parser.add_argument(
        "--sigma", type=float, default=0.1, metavar="THZ",
        help="Gaussian linewidth broadening in THz (default: 0.1; converge this value).",
    )
    parser.add_argument("--no-isotope", action="store_true", help="Disable natural-isotope scattering.")
    parser.add_argument("--overwrite-conversion", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_temperature(meta_file: Path) -> float:
    """Read the target temperature from a native TDEP ``infile.meta`` file."""
    values = [line.split() for line in meta_file.read_text().splitlines() if line.strip()]
    try:
        return float(values[3][0])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Could not read the temperature from {meta_file}.") from error


def validate_fitting_dataset(iteration_dir: Path) -> tuple[int, int]:
    """Check the native TDEP data before invoking ``extract_forceconstants``."""
    missing = [name for name in FIT_INPUTS if not (iteration_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{iteration_dir} is missing native TDEP fitting input(s): {', '.join(missing)}"
        )
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


def read_poscar_lattice(path: Path) -> np.ndarray:
    """Read a POSCAR lattice in Angstrom, including its scale convention."""
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
    if scale < 0.0:
        scale = (-scale / volume) ** (1.0 / 3.0)
    return lattice * scale


def supercell_safe_cutoff(ssposcar: Path) -> float:
    """Return the largest non-aliased real-space cutoff for the supercell."""
    cell = read_poscar_lattice(ssposcar)
    volume = abs(float(np.linalg.det(cell)))
    heights = [volume / np.linalg.norm(np.cross(cell[(axis + 1) % 3], cell[(axis + 2) % 3])) for axis in range(3)]
    return 0.5 * min(heights)


def secondorder_cutoff(config_file: Path) -> float:
    try:
        with config_file.open() as handle:
            value = float(yaml.safe_load(handle)["tdep"]["secondorder_cutoff_A"])
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read tdep.secondorder_cutoff_A from {config_file}.") from error
    if value <= 0.0:
        raise ValueError(f"tdep.secondorder_cutoff_A in {config_file} must be positive.")
    return value


def extract_forceconstants(config_file: Path) -> str:
    """Locate TDEP relative to the original project, not a copied config file."""
    try:
        with config_file.open() as handle:
            bin_dir = Path(yaml.safe_load(handle)["tdep"]["bin_dir"])
    except (KeyError, TypeError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read tdep.bin_dir from {config_file}.") from error
    if bin_dir.is_absolute():
        candidates = [bin_dir / "extract_forceconstants"]
    else:
        # ``config_used.yaml`` is copied into an output directory by
        # run_tdep.py, while bin_dir remains relative to the repository YAML.
        # Prefer the copied-config interpretation only when it actually works.
        candidates = [
            config_file.parent / bin_dir / "extract_forceconstants",
            Path.cwd() / bin_dir / "extract_forceconstants",
        ]
    executable = next(
        (candidate.resolve() for candidate in candidates if candidate.is_file() and candidate.stat().st_mode & 0o111),
        candidates[-1].resolve(),
    )
    return str(executable)


def link(source: Path, target: Path) -> None:
    """Create a verified input link without replacing user data."""
    source = source.resolve()
    if target.is_symlink() and target.resolve() == source:
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to replace existing {target}.")
    target.symlink_to(source)


def fit_force_constants(
    *,
    iteration: Path,
    fit_dir: Path,
    rc2: float,
    rc3: float,
    temperature: float,
    executable: str,
    dry_run: bool,
    stride: int = 1,
) -> tuple[Path, Path]:
    """Fit or reuse the shared native IFC2/IFC3 pair for one iteration."""
    fc2, fc3 = fit_dir / "outfile.forceconstant", fit_dir / "outfile.forceconstant_thirdorder"
    if fc2.is_file() and fc3.is_file():
        print(f"  reuse shared IFCs in {fit_dir}")
        return fc2, fc3
    if fc2.exists() or fc3.exists():
        raise FileExistsError(f"{fit_dir} contains an incomplete IFC pair; refusing to overwrite it.")
    command = [
        executable,
        "--secondorder_cutoff", f"{rc2:.12g}",
        "--thirdorder_cutoff", f"{rc3:.12g}",
        "--temperature", f"{temperature:.12g}",
    ]
    if stride < 1:
        raise ValueError("--stride must be a positive integer.")
    if stride != 1:
        command.extend(["--stride", str(stride)])
    print("+", " ".join(command), f"  # cwd={fit_dir}")
    if dry_run:
        return fc2, fc3
    if not Path(executable).is_file() or not Path(executable).stat().st_mode & 0o111:
        raise FileNotFoundError(f"TDEP executable is not executable: {executable}")
    fit_dir.mkdir(exist_ok=True)
    if iteration.resolve() != fit_dir.resolve():
        for filename in FIT_INPUTS:
            link(iteration / filename, fit_dir / filename)
    subprocess.run(command, cwd=fit_dir, check=True)
    if not fc2.is_file() or not fc3.is_file():
        raise RuntimeError(f"TDEP did not write both IFC2 and IFC3 in {fit_dir}.")
    return fc2, fc3


def require_wte_plugin() -> None:
    """Load WTE and verify that both solver variants were registered.

    Merely finding the ``phono3py.conductivity`` entry point is insufficient:
    phono3py deliberately ignores an entry point that raises while importing.
    Importing and registering it here turns a plugin/API mismatch into an
    immediate, actionable error before an expensive IFC conversion starts.
    """
    try:
        from wte import register
    except ImportError as error:
        raise RuntimeError(
            "The phono3py WTE plugin could not be imported. Install a phono3py-wte "
            "release compatible with the installed phono3py version."
        ) from error

    try:
        register()
        from phono3py.conductivity.factory import _REGISTRY
    except Exception as error:
        raise RuntimeError(
            "The phono3py WTE plugin could not be registered; its API is incompatible "
            "with the installed phono3py version."
        ) from error

    missing = {"wte-rta", "wte-lbte"}.difference(_REGISTRY)
    if missing:
        raise RuntimeError(
            "The phono3py WTE plugin was imported but did not register "
            f"{', '.join(sorted(missing))}."
        )


def supercell_matrix(config_file: Path, override: list[int] | None) -> tuple[int, int, int]:
    if override is not None:
        values = override
    else:
        with config_file.open() as handle:
            config = yaml.safe_load(handle)
        try:
            values = config["tdep"]["supercell_matrix"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Could not read tdep.supercell_matrix from {config_file}.") from error
    if len(values) != 3 or any(int(value) < 1 for value in values):
        raise ValueError("--supercell-matrix must contain three positive diagonal dimensions.")
    return tuple(int(value) for value in values)


def load_force_constants(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        key = "force_constants" if path.name == "fc2.hdf5" else "fc3"
        return np.array(handle[key], dtype=float, order="C")


def sample_fit_and_calculate_wte(args: argparse.Namespace, iteration: Path) -> None:
    """Run the new-configuration path without exposing another CLI."""
    assert args.new_configurations is not None
    from fit_tdep_ifc3 import run_new_configuration_workflow

    run_new_configuration_workflow(
        argparse.Namespace(
            result_dir=iteration,
            config=args.sampling_config.resolve(),
            thirdorder_cutoff=args.thirdorder_cutoff,
            configurations=args.new_configurations,
            secondorder_cutoff=None,
            stride=1,
            qpoint_grid=args.qpoint_grid,
            supercell_matrix=args.supercell_matrix,
            solver=args.solver,
            sigma=args.sigma,
            no_isotope=args.no_isotope,
            overwrite_conversion=args.overwrite_conversion,
            output_dir=None,
            stop_after_labeling=False,
            handoff_dir=None,
            dry_run=args.dry_run,
        )
    )


def calculate_wte(
    *,
    iteration: Path,
    fit_dir: Path,
    fc2_file: Path,
    fc3_file: Path,
    dimensions: tuple[int, int, int],
    qpoint_grid: list[int],
    solver: str,
    sigma: float,
    no_isotope: bool,
    overwrite_conversion: bool,
) -> None:
    """Convert one shared IFC pair if needed and calculate/reuse its WTE kappa."""
    phono3py_ifcs = fit_dir / "phono3py_ifcs"
    result_dir = fit_dir / (
        f"phono3py_wte_{solver}_qg_" + "x".join(map(str, qpoint_grid)) + f"_sigma_{sigma:g}THz"
    )
    convert_tdep_force_constants(
        ucposcar=iteration / "infile.ucposcar",
        ssposcar=iteration / "infile.ssposcar",
        fc2_file=fc2_file,
        fc3_file=fc3_file,
        output_dir=phono3py_ifcs,
        supercell_matrix=dimensions,
        overwrite=overwrite_conversion,
    )
    result_dir.mkdir(exist_ok=True)
    if any(result_dir.glob("kappa-*.hdf5")):
        print(f"  reuse existing WTE result in {result_dir}")
        return
    ph3 = build_phono3py(iteration / "infile.ucposcar", dimensions)
    ph3.fc2 = load_force_constants(phono3py_ifcs / "fc2.hdf5")
    ph3.fc3 = load_force_constants(phono3py_ifcs / "fc3.hdf5")
    ph3.mesh_numbers = np.array(qpoint_grid, dtype=int)
    ph3.sigmas = [sigma]
    ph3.init_phph_interaction()
    ph3.run_phonon_solver()
    with working_directory(result_dir):
        ph3.run_thermal_conductivity(
            is_LBTE=solver == "lbte",
            temperatures=np.array([read_temperature(iteration / "infile.meta")]),
            is_isotope=not no_isotope,
            transport_type="WTE",
            write_kappa=True,
            log_level=1,
        )


def main() -> None:
    args = parse_arguments()
    if args.thirdorder_cutoff <= 0 or args.sigma <= 0 or any(value < 1 for value in args.qpoint_grid):
        raise ValueError("Cutoff, sigma, and q-point-grid dimensions must be positive.")
    if args.new_configurations is not None and args.new_configurations < 1:
        raise ValueError("--new-configurations must be a positive integer.")
    input_root = args.input_root.resolve()
    config_file = input_root / "config_used.yaml"
    dimensions = supercell_matrix(config_file, args.supercell_matrix)
    rc2 = secondorder_cutoff(config_file)
    # An existing IFC pair can be converted and used for WTE on a machine
    # without TDEP installed, so only validate the executable if fitting is
    # actually necessary.
    extract = extract_forceconstants(config_file)

    # A dry run is intended to be usable on a machine that has TDEP but not
    # phono3py/WTE installed, so defer the plugin import until real work starts.
    if not args.dry_run:
        require_wte_plugin()

    for number in args.iterations:
        iteration = input_root / f"iteration_{number:02d}"
        if args.new_configurations is not None:
            print(f"iteration {number}: generate and fit {args.new_configurations} new configurations")
            sample_fit_and_calculate_wte(args, iteration)
            continue
        n_atoms, n_configurations = validate_fitting_dataset(iteration)
        temperature = read_temperature(iteration / "infile.meta")
        safe_cutoff = supercell_safe_cutoff(iteration / "infile.ssposcar")
        if max(rc2, args.thirdorder_cutoff) > safe_cutoff + 1.0e-8:
            raise ValueError(
                f"iteration {number}: requested cutoff exceeds its safe maximum ({safe_cutoff:.3f} A)."
            )
        fit_dir = iteration / f"ifc3_rc3_{args.thirdorder_cutoff:g}A"
        print(
            f"iteration {number}: {n_configurations} configurations, {temperature:g} K; "
            f"safe cutoff {safe_cutoff:.3f} A"
        )
        fc2_file, fc3_file = fit_force_constants(
            iteration=iteration,
            fit_dir=fit_dir,
            rc2=rc2,
            rc3=args.thirdorder_cutoff,
            temperature=temperature,
            executable=extract,
            dry_run=args.dry_run,
        )
        print(f"  phono3py WTE: {args.solver.upper()}, q grid {args.qpoint_grid}, sigma {args.sigma:g} THz")
        if args.dry_run:
            continue
        calculate_wte(
            iteration=iteration,
            fit_dir=fit_dir,
            fc2_file=fc2_file,
            fc3_file=fc3_file,
            dimensions=dimensions,
            qpoint_grid=args.qpoint_grid,
            solver=args.solver,
            sigma=args.sigma,
            no_isotope=args.no_isotope,
            overwrite_conversion=args.overwrite_conversion,
        )


if __name__ == "__main__":
    main()
