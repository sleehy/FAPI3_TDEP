"""Convert TDEP real-space IFC files to compact phono3py HDF5 force constants.

TDEP stores IFCs as primitive-cell atoms plus lattice translations, whereas
phono3py uses a dense compact array indexed by a specified supercell.  This
module maps the TDEP representation onto phono3py's *own* atom ordering, so
the resulting ``fc2.hdf5`` and ``fc3.hdf5`` can be reused for every q mesh.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from ase.io import read
from phonopy.structure.atoms import PhonopyAtoms
from phono3py import Phono3py


def build_phono3py(ucposcar: Path, supercell_matrix: tuple[int, int, int]) -> Phono3py:
    """Create the phono3py cell with the TDEP unit-cell atom ordering intact."""
    atoms = read(ucposcar, format="vasp")
    unitcell = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(wrap=True),
    )
    return Phono3py(
        unitcell,
        supercell_matrix=np.diag(supercell_matrix),
        primitive_matrix=np.eye(3),
        symprec=1.0e-5,
    )


def _readline(handle, path: Path) -> list[str]:
    while line := handle.readline():
        fields = line.split()
        if fields:
            return fields
    raise ValueError(f"Unexpected end of TDEP force-constant file: {path}")


def _integer(handle, path: Path) -> int:
    return int(_readline(handle, path)[0])


def _vector(handle, path: Path) -> np.ndarray:
    fields = _readline(handle, path)
    if len(fields) < 3:
        raise ValueError(f"Expected a three-vector in {path}.")
    return np.array([float(value) for value in fields[:3]], dtype=float)


def _translation(vector: np.ndarray, path: Path) -> tuple[int, int, int]:
    rounded = np.rint(vector).astype(int)
    if not np.allclose(vector, rounded, atol=1.0e-7, rtol=0.0):
        raise ValueError(f"TDEP lattice vector {vector} in {path} is not integral.")
    return tuple(int(value) for value in rounded)


def _supercell_index_map(
    ph3: Phono3py, ucposcar: Path, ssposcar: Path, dimensions: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Map (unit-cell atom, lattice translation) to phono3py supercell index."""
    unit_atoms = read(ucposcar, format="vasp")
    tdep_supercell = read(ssposcar, format="vasp")
    if len(tdep_supercell) != len(unit_atoms) * int(np.prod(dimensions)):
        raise ValueError("infile.ssposcar atom count does not match --supercell-matrix.")
    if not np.allclose(tdep_supercell.cell.array, ph3.supercell.cell, atol=1.0e-5, rtol=0.0):
        raise ValueError("infile.ssposcar cell does not match the requested phono3py supercell.")

    unit_positions = unit_atoms.get_scaled_positions(wrap=True)
    super_positions = np.asarray(ph3.supercell.scaled_positions) % 1.0
    unit_symbols = unit_atoms.get_chemical_symbols()
    super_symbols = list(ph3.supercell.symbols)
    mapping = np.full((len(unit_atoms), *dimensions), -1, dtype=int)
    used: set[int] = set()
    dimensions_array = np.asarray(dimensions, dtype=float)
    for atom, position in enumerate(unit_positions):
        candidates = np.array([index for index, symbol in enumerate(super_symbols) if symbol == unit_symbols[atom]])
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                for k in range(dimensions[2]):
                    expected = (position + np.array([i, j, k])) / dimensions_array
                    difference = super_positions[candidates] - expected
                    difference -= np.rint(difference)
                    errors = np.max(np.abs(difference), axis=1)
                    local = int(np.argmin(errors))
                    index = int(candidates[local])
                    if errors[local] > 2.0e-5 or index in used:
                        raise ValueError("Could not construct a one-to-one TDEP/phono3py supercell atom map.")
                    mapping[atom, i, j, k] = index
                    used.add(index)
    if len(used) != len(super_positions):
        raise ValueError("TDEP/phono3py supercell atom map is incomplete.")

    # Phono3py's primitive order can differ from the POSCAR order even with an
    # identity primitive matrix. Match it explicitly before filling compact FCs.
    primitive_positions = np.asarray(ph3.primitive.scaled_positions) % 1.0
    primitive_symbols = list(ph3.primitive.symbols)
    primitive_to_unit = np.full(len(primitive_positions), -1, dtype=int)
    for primitive_atom, position in enumerate(primitive_positions):
        candidates = np.array([index for index, symbol in enumerate(unit_symbols) if symbol == primitive_symbols[primitive_atom]])
        difference = unit_positions[candidates] - position
        difference -= np.rint(difference)
        errors = np.max(np.abs(difference), axis=1)
        local = int(np.argmin(errors))
        if errors[local] > 2.0e-5:
            raise ValueError("Could not map phono3py primitive atoms to infile.ucposcar.")
        primitive_to_unit[primitive_atom] = int(candidates[local])
    return mapping, primitive_to_unit


def _target(mapping: np.ndarray, atom: int, translation: tuple[int, int, int], dimensions: tuple[int, int, int]) -> int:
    return int(mapping[atom, *(translation[axis] % dimensions[axis] for axis in range(3))])


def parse_fc2(path: Path, mapping: np.ndarray, primitive_to_unit: np.ndarray, dimensions: tuple[int, int, int]) -> tuple[np.ndarray, float]:
    n_primitive, n_supercell = len(primitive_to_unit), mapping.size
    fc2 = np.zeros((n_primitive, n_supercell, 3, 3), dtype=float)
    with path.open() as handle:
        n_atoms = _integer(handle, path)
        cutoff = float(_readline(handle, path)[0])
        if n_atoms != n_primitive:
            raise ValueError(f"{path} has {n_atoms} atoms but the unit cell has {n_primitive}.")
        unit_to_primitive = {unit: primitive for primitive, unit in enumerate(primitive_to_unit)}
        for unit_atom in range(n_atoms):
            count = _integer(handle, path)
            primitive_atom = unit_to_primitive[unit_atom]
            for _ in range(count):
                neighbour = _integer(handle, path) - 1
                translation = _translation(_vector(handle, path), path)
                matrix = np.array([_vector(handle, path) for _ in range(3)], dtype=float)
                fc2[primitive_atom, _target(mapping, neighbour, translation, dimensions)] += matrix
    return fc2, cutoff


def parse_fc3(path: Path, mapping: np.ndarray, primitive_to_unit: np.ndarray, dimensions: tuple[int, int, int]) -> tuple[np.ndarray, float]:
    n_primitive, n_supercell = len(primitive_to_unit), mapping.size
    # phono3py's compact FC3 is dense by design. For this 24-atom/3x3x2 case
    # it occupies about 1 GiB in memory before HDF5 compression.
    fc3 = np.zeros((n_primitive, n_supercell, n_supercell, 3, 3, 3), dtype=float)
    with path.open() as handle:
        n_atoms = _integer(handle, path)
        cutoff = float(_readline(handle, path)[0])
        if n_atoms != n_primitive:
            raise ValueError(f"{path} has {n_atoms} atoms but the unit cell has {n_primitive}.")
        unit_to_primitive = {unit: primitive for primitive, unit in enumerate(primitive_to_unit)}
        for unit_atom in range(n_atoms):
            count = _integer(handle, path)
            primitive_atom = unit_to_primitive[unit_atom]
            for _ in range(count):
                first, second, third = (_integer(handle, path) - 1 for _ in range(3))
                first_translation = _translation(_vector(handle, path), path)
                second_translation = _translation(_vector(handle, path), path)
                third_translation = _translation(_vector(handle, path), path)
                if first != unit_atom or first_translation != (0, 0, 0):
                    raise ValueError(
                        f"{path} contains a non-origin first atom; this converter requires TDEP's standard anchored format."
                    )
                tensor = np.empty((3, 3, 3), dtype=float)
                for i in range(3):
                    for j in range(3):
                        tensor[i, j] = _vector(handle, path)
                fc3[primitive_atom, _target(mapping, second, second_translation, dimensions),
                    _target(mapping, third, third_translation, dimensions)] += tensor
    return fc3, cutoff


def convert_tdep_force_constants(
    *, ucposcar: Path, ssposcar: Path, fc2_file: Path, fc3_file: Path, output_dir: Path,
    supercell_matrix: tuple[int, int, int], overwrite: bool = False,
) -> tuple[Path, Path]:
    """Convert one fitted TDEP IFC2/IFC3 pair to reusable phono3py HDF5 files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fc2_output, fc3_output = output_dir / "fc2.hdf5", output_dir / "fc3.hdf5"
    if fc2_output.exists() or fc3_output.exists():
        if not overwrite and fc2_output.is_file() and fc3_output.is_file():
            return fc2_output, fc3_output
        raise FileExistsError(f"Refusing to replace existing phono3py IFC files in {output_dir}.")
    ph3 = build_phono3py(ucposcar, supercell_matrix)
    mapping, primitive_to_unit = _supercell_index_map(ph3, ucposcar, ssposcar, supercell_matrix)
    fc2, cutoff2 = parse_fc2(fc2_file, mapping, primitive_to_unit, supercell_matrix)
    fc3, cutoff3 = parse_fc3(fc3_file, mapping, primitive_to_unit, supercell_matrix)
    with h5py.File(fc2_output, "w") as handle:
        handle.create_dataset("force_constants", data=fc2, compression="gzip")
        handle.create_dataset("p2s_map", data=ph3.primitive.p2s_map)
        handle.create_dataset("cutoff", data=cutoff2)
        handle.create_dataset("physical_unit", data=np.bytes_("eV/angstrom^2"))
    with h5py.File(fc3_output, "w") as handle:
        handle.create_dataset("fc3", data=fc3, compression="gzip")
        handle.create_dataset("p2s_map", data=ph3.primitive.p2s_map)
        handle.create_dataset("fc3_cutoff", data=cutoff3)
    return fc2_output, fc3_output
