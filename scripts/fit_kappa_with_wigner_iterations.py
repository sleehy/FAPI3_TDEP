#!/usr/bin/env python3
"""Calculate iteration-resolved Wigner thermal conductivity with phono3py.

This script deliberately does *not* fit force constants. It reuses the joint
TDEP IFC2/IFC3 pair written by ``fit_ifc3_kappa_iterations.py --fit-only`` (or
by that script's normal TDEP-kappa run), converts it once to phono3py HDF5,
and evaluates the phono3py-WTE solver on a chosen q mesh.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import yaml

from fit_tdep_ifc3 import read_temperature
from tdep_to_phono3py import build_phono3py, convert_tdep_force_constants


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


def require_wte_plugin() -> None:
    plugins = importlib.metadata.entry_points(group="phono3py.conductivity")
    if not any(entry.name.upper() == "WTE" for entry in plugins):
        raise RuntimeError(
            "phono3py-wte is required for the official phono3py Wigner solver. Install it with "
            "`python -m pip install git+https://github.com/MSimoncelli/phono3py-wte.git`."
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


def main() -> None:
    args = parse_arguments()
    if args.thirdorder_cutoff <= 0 or args.sigma <= 0 or any(value < 1 for value in args.qpoint_grid):
        raise ValueError("Cutoff, sigma, and q-point-grid dimensions must be positive.")
    require_wte_plugin()
    input_root = args.input_root.resolve()
    dimensions = supercell_matrix(input_root / "config_used.yaml", args.supercell_matrix)

    for number in args.iterations:
        iteration = input_root / f"iteration_{number:02d}"
        fit_dir = iteration / f"ifc3_rc3_{args.thirdorder_cutoff:g}A"
        fc2_file, fc3_file = fit_dir / "outfile.forceconstant", fit_dir / "outfile.forceconstant_thirdorder"
        if not fc2_file.is_file() or not fc3_file.is_file():
            raise FileNotFoundError(
                f"iteration {number} has no shared TDEP IFC2/IFC3 pair in {fit_dir}. Run:\n"
                f"  python scripts/fit_ifc3_kappa_iterations.py --input-root {input_root} "
                f"--iterations {number} --thirdorder-cutoff {args.thirdorder_cutoff:g} --fit-only"
            )
        phono3py_ifcs = fit_dir / "phono3py_ifcs"
        result_dir = fit_dir / (
            f"phono3py_wte_{args.solver}_qg_" + "x".join(map(str, args.qpoint_grid))
            + f"_sigma_{args.sigma:g}THz"
        )
        print(f"iteration {number}: shared IFCs {fc2_file.name}, {fc3_file.name}")
        print(f"  phono3py WTE: {args.solver.upper()}, q grid {args.qpoint_grid}, sigma {args.sigma:g} THz")
        if args.dry_run:
            continue

        convert_tdep_force_constants(
            ucposcar=iteration / "infile.ucposcar",
            ssposcar=iteration / "infile.ssposcar",
            fc2_file=fc2_file,
            fc3_file=fc3_file,
            output_dir=phono3py_ifcs,
            supercell_matrix=dimensions,
            overwrite=args.overwrite_conversion,
        )
        result_dir.mkdir(exist_ok=True)
        if any(result_dir.glob("kappa-*.hdf5")):
            print(f"  reuse existing WTE result in {result_dir}")
            continue

        ph3 = build_phono3py(iteration / "infile.ucposcar", dimensions)
        ph3.fc2 = load_force_constants(phono3py_ifcs / "fc2.hdf5")
        ph3.fc3 = load_force_constants(phono3py_ifcs / "fc3.hdf5")
        ph3.mesh_numbers = np.array(args.qpoint_grid, dtype=int)
        ph3.sigmas = [args.sigma]
        ph3.init_phph_interaction()
        ph3.run_phonon_solver()
        with working_directory(result_dir):
            ph3.run_thermal_conductivity(
                is_LBTE=args.solver == "lbte",
                temperatures=np.array([read_temperature(iteration / "infile.meta")]),
                is_isotope=not args.no_isotope,
                transport_type="WTE",
                write_kappa=True,
                log_level=1,
            )


if __name__ == "__main__":
    main()
