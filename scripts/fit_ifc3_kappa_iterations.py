#!/usr/bin/env python3
"""Fit IFC2+IFC3 and calculate TDEP thermal conductivity for iterations.

The input directories must already contain TDEP's native fitting dataset
(``infile.ucposcar``, ``infile.ssposcar``, ``infile.meta``, ``infile.stat``,
``infile.positions``, and ``infile.forces``).  No configurations are sampled
or relabelled.  Results are written below each iteration, leaving its original
FC2 and all input files unchanged.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml

# Keep the IFC3-data checks and the no-aliasing cutoff check identical to the
# established IFC3 workflow.  The sampling part of fit_tdep_ifc3.py is not
# needed: these iterations already contain labelled TDEP datasets.
from fit_tdep_ifc3 import (
    FIT_INPUTS,
    cutoff_from_config,
    read_temperature,
    supercell_safe_cutoff,
    validate_fitting_dataset,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("ifc3_iter12_16"))
    parser.add_argument(
        "--config", type=Path, default=Path("tdep_tetragonal.yaml"),
        help="Repository TDEP YAML used to locate the official executables.",
    )
    parser.add_argument("--iterations", type=int, nargs="+", default=[12, 13, 14, 15, 16])
    parser.add_argument("--thirdorder-cutoff", required=True, type=float, metavar="ANGSTROM")
    parser.add_argument(
        "--qpoint-grid", type=int, nargs=3, metavar=("N1", "N2", "N3"),
        help="Common BZ q mesh for every iteration (converge this independently).",
    )
    parser.add_argument(
        "--secondorder-cutoff", type=float, metavar="ANGSTROM",
        help="Default: read tdep.secondorder_cutoff_A from <input-root>/config_used.yaml.",
    )
    parser.add_argument("--integrationtype", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--mpi-ranks", type=int, default=1, help="Use mpirun -np N when N > 1.")
    parser.add_argument(
        "--fit-only", action="store_true",
        help="Only create/reuse the shared TDEP IFC2+IFC3 pair; do not calculate TDEP kappa.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def executable(config: dict, config_file: Path, name: str) -> str:
    try:
        bin_dir = Path(config["tdep"]["bin_dir"])
    except KeyError as error:
        raise ValueError(f"{config_file} is missing tdep.bin_dir.") from error
    path = bin_dir if bin_dir.is_absolute() else config_file.parent / bin_dir
    path = (path / name).resolve()
    if not path.is_file() or not path.stat().st_mode & 0o111:
        raise FileNotFoundError(f"TDEP executable is not executable: {path}")
    return str(path)


def link(source: Path, target: Path) -> None:
    """Create a verified relative input link without replacing user data."""
    source = source.resolve()
    if target.is_symlink() and target.resolve() == source:
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to replace existing {target}.")
    target.symlink_to(source)


def command_prefix(ranks: int) -> list[str]:
    return [] if ranks == 1 else ["mpirun", "-np", str(ranks)]


def run(command: list[str], cwd: Path, dry_run: bool) -> None:
    print("+", " ".join(command), f"  # cwd={cwd}")
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    args = parse_arguments()
    if args.thirdorder_cutoff <= 0 or args.mpi_ranks < 1:
        raise ValueError("Cutoff and MPI ranks must be positive.")
    if not args.fit_only and (args.qpoint_grid is None or any(point < 1 for point in args.qpoint_grid)):
        raise ValueError("--qpoint-grid must contain three positive integers unless --fit-only is used.")

    input_root = args.input_root.resolve()
    source_config_file = input_root / "config_used.yaml"
    with source_config_file.open() as handle:
        source_config = yaml.safe_load(handle)
    config_file = args.config.resolve()
    with config_file.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{config_file} must contain a YAML mapping.")
    if not isinstance(source_config, dict):
        raise ValueError(f"{source_config_file} must contain a YAML mapping.")
    rc2 = args.secondorder_cutoff if args.secondorder_cutoff is not None else cutoff_from_config(source_config_file)
    if rc2 <= 0:
        raise ValueError("--secondorder-cutoff must be positive.")
    extract = executable(config, config_file, "extract_forceconstants")
    kappa = executable(config, config_file, "thermal_conductivity_2023")
    prefix = command_prefix(args.mpi_ranks)

    for number in args.iterations:
        iteration = input_root / f"iteration_{number:02d}"
        n_atoms, n_configurations = validate_fitting_dataset(iteration)
        temperature = read_temperature(iteration / "infile.meta")
        safe_cutoff = supercell_safe_cutoff(iteration / "infile.ssposcar")
        if max(rc2, args.thirdorder_cutoff) > safe_cutoff + 1.0e-8:
            raise ValueError(
                f"iteration {number}: requested cutoff exceeds its safe maximum "
                f"({safe_cutoff:.3f} A)."
            )
        fit_dir = iteration / f"ifc3_rc3_{args.thirdorder_cutoff:g}A"
        if not args.dry_run:
            fit_dir.mkdir(exist_ok=True)
            for filename in FIT_INPUTS:
                link(iteration / filename, fit_dir / filename)

        fc2, fc3 = fit_dir / "outfile.forceconstant", fit_dir / "outfile.forceconstant_thirdorder"
        print(
            f"iteration {number}: {n_configurations} configurations, {temperature:g} K; "
            f"safe cutoff {safe_cutoff:.3f} A"
        )
        if not fc3.is_file():
            run(prefix + [extract, "--secondorder_cutoff", f"{rc2:.12g}",
                          "--thirdorder_cutoff", f"{args.thirdorder_cutoff:.12g}",
                          "--temperature", f"{temperature:.12g}"], fit_dir, args.dry_run)
        else:
            print(f"  reuse {fc3}")
        if not args.dry_run and not (fc2.is_file() and fc3.is_file()):
            raise RuntimeError(f"iteration {number}: extract_forceconstants did not write both FC2 and IFC3.")

        if args.fit_only:
            continue

        assert args.qpoint_grid is not None
        kappa_dir = fit_dir / ("kappa_qg_" + "x".join(map(str, args.qpoint_grid)))
        output = kappa_dir / "outfile.thermal_conductivity"
        if output.is_file():
            print(f"  reuse {output}")
            continue
        if not args.dry_run:
            kappa_dir.mkdir(exist_ok=True)
            link(iteration / "infile.ucposcar", kappa_dir / "infile.ucposcar")
            link(fc2, kappa_dir / "infile.forceconstant")
            link(fc3, kappa_dir / "infile.forceconstant_thirdorder")
        run(prefix + [kappa, "--qpoint_grid", *map(str, args.qpoint_grid),
                      "--temperature", f"{temperature:.12g}",
                      "--integrationtype", str(args.integrationtype)], kappa_dir, args.dry_run)


if __name__ == "__main__":
    main()
