#!/usr/bin/env python3
"""Move minimal native TDEP fitting datasets through a Git-tracked handoff.

Use this script when SevenNet labels are evaluated on Colab and
``extract_forceconstants`` is run on a local machine. A handoff contains only
the six native TDEP fitting inputs, a small integrity manifest, and the fitted
FC2 or IFC3 force-constant file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


FIT_INPUTS = (
    "infile.ucposcar", "infile.ssposcar", "infile.meta", "infile.stat",
    "infile.positions", "infile.forces",
)
MANIFEST = "manifest.json"
FC2_OUTPUT = "outfile.forceconstant"
IFC3_OUTPUT = "outfile.forceconstant_thirdorder"
FORMAT = "tdep-forceconstant-handoff-v1"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, int | str]:
    return {"sha256": file_digest(path), "bytes": path.stat().st_size}


def read_config(path: Path) -> dict:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return config


def config_value_path(config: dict, config_file: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_file.parent / path


def manifest_path(handoff_dir: Path) -> Path:
    return handoff_dir / MANIFEST


def load_manifest(handoff_dir: Path) -> dict:
    path = manifest_path(handoff_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Missing handoff manifest: {path}")
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Could not parse {path}.") from error
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} manifest.")
    return manifest


def validate_inputs(handoff_dir: Path, manifest: dict) -> None:
    records = manifest.get("inputs")
    if not isinstance(records, dict):
        raise ValueError("Handoff manifest has no input checksums.")
    for name in FIT_INPUTS:
        path = handoff_dir / name
        expected = records.get(name)
        if not path.is_file() or not isinstance(expected, dict):
            raise FileNotFoundError(f"Handoff is missing required input: {path}")
        if file_record(path) != expected:
            raise ValueError(f"Handoff input failed integrity check: {path}")


def export_fitting_inputs(
    iteration_dir: Path,
    handoff_dir: Path,
    config_file: Path,
    output_directory: Path,
    iteration: int,
    *,
    kind: str = "fc2",
    fit_arguments: dict[str, float | int] | None = None,
) -> Path:
    """Copy only fitting inputs into a new Git handoff directory."""
    if kind not in {"fc2", "ifc3"}:
        raise ValueError(f"Unsupported handoff kind: {kind}")
    missing = [name for name in FIT_INPUTS if not (iteration_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Cannot export TDEP handoff; {iteration_dir} is missing: {', '.join(missing)}"
        )
    if handoff_dir.exists() and any(handoff_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing handoff {handoff_dir}. "
            "Commit/push it, or choose a new --handoff-dir."
        )
    handoff_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for name in FIT_INPUTS:
        source, target = iteration_dir / name, handoff_dir / name
        shutil.copy2(source, target)
        records[name] = file_record(target)
    manifest = {
        "format": FORMAT,
        "kind": kind,
        "state": "ready_for_local_fit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iteration": iteration,
        "source_output_directory": str(output_directory),
        "config_sha256": file_digest(config_file),
        "inputs": records,
    }
    if fit_arguments is not None:
        manifest["fit_arguments"] = fit_arguments
    manifest_path(handoff_dir).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return handoff_dir


def tdep_executable(config: dict, config_file: Path) -> str:
    try:
        executable = config_value_path(config, config_file, str(config["tdep"]["bin_dir"])) / "extract_forceconstants"
    except KeyError as error:
        raise ValueError("Config is missing tdep.bin_dir.") from error
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"TDEP executable not found or not executable: {executable}")
    return str(executable)


def fit_handoff(handoff_dir: Path, config_file: Path, dry_run: bool) -> None:
    handoff_dir, config_file = handoff_dir.resolve(), config_file.resolve()
    manifest = load_manifest(handoff_dir)
    if manifest.get("kind") != "fc2":
        raise ValueError("This is not an FC2 handoff; use `tdep_handoff.py fit-ifc3` for IFC3.")
    validate_inputs(handoff_dir, manifest)
    if manifest.get("config_sha256") != file_digest(config_file):
        raise ValueError(
            "The active config does not exactly match the config used to create this handoff. "
            "Use the same committed YAML on both machines."
        )
    config = read_config(config_file)
    try:
        cutoff = float(config["tdep"]["secondorder_cutoff_A"])
        temperature = float(config["tdep"]["temperature_K"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Config needs numeric tdep.secondorder_cutoff_A and tdep.temperature_K.") from error
    command = [tdep_executable(config, config_file), "--secondorder_cutoff", str(cutoff),
               "--temperature", str(temperature)]
    print("+", " ".join(command))
    if dry_run:
        return
    forceconstant = handoff_dir / FC2_OUTPUT
    if forceconstant.is_file():
        print(f"FC2 already exists; reusing {forceconstant}")
        return
    subprocess.run(command, cwd=handoff_dir, check=True)
    if not forceconstant.is_file():
        raise RuntimeError("TDEP completed without writing outfile.forceconstant.")
    manifest["state"] = "fitted_locally"
    manifest["fitted_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["forceconstant"] = file_record(forceconstant)
    manifest_path(handoff_dir).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {forceconstant}")


def import_forceconstant(handoff_dir: Path, config_file: Path) -> Path:
    handoff_dir, config_file = handoff_dir.resolve(), config_file.resolve()
    manifest = load_manifest(handoff_dir)
    if manifest.get("kind") != "fc2":
        raise ValueError("This is not an FC2 handoff; use `tdep_handoff.py import-ifc3` for IFC3.")
    validate_inputs(handoff_dir, manifest)
    if manifest.get("state") != "fitted_locally":
        raise ValueError(f"Handoff state is {manifest.get('state')!r}, not 'fitted_locally'.")
    if manifest.get("config_sha256") != file_digest(config_file):
        raise ValueError("The active config does not exactly match the config used to create this handoff.")
    source, expected = handoff_dir / FC2_OUTPUT, manifest.get("forceconstant")
    if not source.is_file() or not isinstance(expected, dict) or file_record(source) != expected:
        raise ValueError(f"Handoff FC2 failed integrity check: {source}")
    config = read_config(config_file)
    output_dir = config_value_path(config, config_file, str(config["output"]["directory"]))
    try:
        iteration = int(manifest["iteration"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Handoff manifest has an invalid iteration number.") from error
    destination = output_dir / f"iteration_{iteration:02d}" / FC2_OUTPUT
    # The fitting data stays on Colab and is intentionally not downloaded
    # again. Require that retained local copy to be identical before pairing
    # it with a returned FC2.
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"Missing Colab-side labelled iteration directory: {destination.parent}. "
            "Keep the ignored tdep_tetragonal_* result directory between handoffs."
        )
    for name, expected_input in manifest["inputs"].items():
        local_input = destination.parent / name
        if not local_input.is_file() or file_record(local_input) != expected_input:
            raise ValueError(
                f"Local labelled input does not match this handoff: {local_input}. "
                "Do not import an FC2 into a different Colab run directory."
            )
    if destination.is_file():
        if file_digest(destination) == expected["sha256"]:
            print(f"FC2 already imported: {destination}")
            return destination
        raise FileExistsError(f"Refusing to replace a different local FC2: {destination}")
    shutil.copy2(source, destination)
    print(f"Imported {source} -> {destination}")
    return destination


def ifc3_fit_arguments(manifest: dict) -> tuple[float, float, float, int]:
    """Read the immutable third-order fitting parameters from a handoff."""
    try:
        arguments = manifest["fit_arguments"]
        secondorder = float(arguments["secondorder_cutoff_A"])
        thirdorder = float(arguments["thirdorder_cutoff_A"])
        temperature = float(arguments["temperature_K"])
        stride = int(arguments["stride"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("IFC3 handoff has invalid fitting arguments.") from error
    if secondorder <= 0.0 or thirdorder <= 0.0 or stride < 1:
        raise ValueError("IFC3 handoff has non-positive cutoffs or stride.")
    return secondorder, thirdorder, temperature, stride


def fit_ifc3_handoff(handoff_dir: Path, config_file: Path, dry_run: bool) -> None:
    """Run the memory-intensive joint IFC2/IFC3 fit on the local machine."""
    handoff_dir, config_file = handoff_dir.resolve(), config_file.resolve()
    manifest = load_manifest(handoff_dir)
    if manifest.get("kind") != "ifc3":
        raise ValueError("This is not an IFC3 handoff; use `tdep_handoff.py fit` for FC2.")
    validate_inputs(handoff_dir, manifest)
    if manifest.get("config_sha256") != file_digest(config_file):
        raise ValueError("The active config does not exactly match the config used to create this handoff.")
    secondorder, thirdorder, temperature, stride = ifc3_fit_arguments(manifest)
    command = [
        tdep_executable(read_config(config_file), config_file),
        "--secondorder_cutoff", f"{secondorder:.12g}",
        "--thirdorder_cutoff", f"{thirdorder:.12g}",
        "--temperature", f"{temperature:.12g}",
        "--stride", str(stride),
    ]
    print("+", " ".join(command))
    if dry_run:
        return
    thirdorder_output = handoff_dir / IFC3_OUTPUT
    if thirdorder_output.is_file():
        print(f"IFC3 already exists; reusing {thirdorder_output}")
        return
    subprocess.run(command, cwd=handoff_dir, check=True)
    if not thirdorder_output.is_file():
        raise RuntimeError("TDEP completed without writing outfile.forceconstant_thirdorder.")
    manifest["state"] = "fitted_locally"
    manifest["fitted_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["thirdorder_forceconstant"] = file_record(thirdorder_output)
    manifest_path(handoff_dir).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {thirdorder_output}")


def import_ifc3_forceconstant(handoff_dir: Path, config_file: Path, output_dir: Path) -> Path:
    """Return only the fitted IFC3 file to the reviewed Colab work directory."""
    handoff_dir, config_file, output_dir = handoff_dir.resolve(), config_file.resolve(), output_dir.resolve()
    manifest = load_manifest(handoff_dir)
    if manifest.get("kind") != "ifc3":
        raise ValueError("This is not an IFC3 handoff; use `tdep_handoff.py import-fc` for FC2.")
    validate_inputs(handoff_dir, manifest)
    if manifest.get("state") != "fitted_locally":
        raise ValueError(f"Handoff state is {manifest.get('state')!r}, not 'fitted_locally'.")
    if manifest.get("config_sha256") != file_digest(config_file):
        raise ValueError("The active config does not exactly match the config used to create this handoff.")
    source, expected = handoff_dir / IFC3_OUTPUT, manifest.get("thirdorder_forceconstant")
    if not source.is_file() or not isinstance(expected, dict) or file_record(source) != expected:
        raise ValueError(f"Handoff IFC3 failed integrity check: {source}")
    if not output_dir.is_dir():
        raise FileNotFoundError(
            f"Missing Colab-side reviewed IFC3 directory: {output_dir}. "
            "Keep the ignored IFC3 work directory between handoffs."
        )
    for name, expected_input in manifest["inputs"].items():
        local_input = output_dir / name
        if not local_input.is_file() or file_record(local_input) != expected_input:
            raise ValueError(
                f"Local reviewed IFC3 input does not match this handoff: {local_input}. "
                "Do not import into a different IFC3 work directory."
            )
    destination = output_dir / IFC3_OUTPUT
    if destination.is_file():
        if file_digest(destination) == expected["sha256"]:
            print(f"IFC3 already imported: {destination}")
            return destination
        raise FileExistsError(f"Refusing to replace a different local IFC3: {destination}")
    shutil.copy2(source, destination)
    print(f"Imported {source} -> {destination}")
    return destination


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("fit", "Run local FC2 extract_forceconstants in a checked-out handoff."),
        ("import-fc", "Copy a locally fitted FC2 into the ignored run directory."),
        ("fit-ifc3", "Run local IFC2+IFC3 extract_forceconstants in a checked-out handoff."),
        ("import-ifc3", "Copy a locally fitted IFC3 into the reviewed IFC3 work directory."),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--handoff-dir", required=True, type=Path)
        command.add_argument("--config", default=Path("tdep_tetragonal.yaml"), type=Path)
        if name in {"fit", "fit-ifc3"}:
            command.add_argument("--dry-run", action="store_true")
        if name == "import-ifc3":
            command.add_argument("--output-dir", required=True, type=Path,
                                 help="Existing reviewed Colab IFC3 work directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.command == "fit":
        fit_handoff(args.handoff_dir, args.config, args.dry_run)
    elif args.command == "import-fc":
        import_forceconstant(args.handoff_dir, args.config)
    elif args.command == "fit-ifc3":
        fit_ifc3_handoff(args.handoff_dir, args.config, args.dry_run)
    else:
        import_ifc3_forceconstant(args.handoff_dir, args.config, args.output_dir)


if __name__ == "__main__":
    main()
