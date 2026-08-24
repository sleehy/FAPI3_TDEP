# FAPbI3 TDEP finite-temperature phonons

Configuration-driven stochastic TDEP phonon renormalization for tetragonal FAPbI3. The workflow uses the native TDEP executables for quantum canonical configuration generation, force-constant fitting, and phonon dispersions, with a fine-tuned SevenNet potential for energy and force evaluation.

## Repository contents

- `tetragonal/CONTCAR_tet`: tetragonal FAPbI3 input structure.
- `orthorhombic/CONTCAR`: companion orthorhombic structure.
- `checkpoint_fine_tuned_al_round1.pth`: SevenNet model checkpoint used by the workflow.
- `external/tdep`: pinned [official TDEP](https://github.com/tdep-developers/tdep) Git submodule.
- `tdep_tetragonal.yaml`: all run parameters.
- `scripts/run_tdep.py`: self-consistent TDEP workflow.

## Default calculation

The supplied configuration runs at 300 K with the following settings:

- 3 × 3 × 2 supercell (432 atoms for the supplied 24-atom tetragonal cell)
- Quantum canonical statistics (`canonical_configuration --quantum`), including zero-point motion
- 200 stochastic configurations per TDEP iteration
- SevenNet energy and force calculations using `checkpoint_fine_tuned_al_round1.pth`
- Native TDEP `extract_forceconstants` fitting and `phonon_dispersion_relations` on the tetragonal Γ–X–M–Γ–Z–R–A–Z path

## Setup and run

Clone with the pinned official TDEP source, then build it locally. TDEP is native Fortran software, so its build requires a Fortran compiler, BLAS/LAPACK, FFTW, MPI, and HDF5 with Fortran support. The provided setup script follows TDEP's official Conda instructions.

```bash
git clone --recurse-submodules https://github.com/sleehy/FAPI3_TDEP.git
cd FAPI3_TDEP
conda create -n fapi3-tdep -c conda-forge python=3.11 gfortran openmpi-mpifort scalapack fftw hdf5
conda activate fapi3-tdep
export TDEP_PREFIX="$CONDA_PREFIX"
bash scripts/setup_tdep.sh
python -m pip install -r requirements.txt
```

`setup_tdep.sh` applies the included compatibility patch before building. It relaxes TDEP's internal symmetry tolerance from `1e-5` to `1e-4` Å (`lo_sqtol` from `1e-10` to `1e-8`) to avoid the known `Bad operation singlets` numerical failure on some x86/BLAS builds. This is a local build patch; the tracked TDEP submodule itself remains pinned to the official revision. If TDEP was already built before pulling this change, run `bash scripts/setup_tdep.sh --rebuild`.

For an existing clone that lacks the submodule, run:

```bash
git submodule update --init --recursive
export TDEP_PREFIX="$CONDA_PREFIX"
bash scripts/setup_tdep.sh
```

Then validate and run the calculation:

```bash
python scripts/run_tdep.py --config tdep_tetragonal.yaml --dry-run
python scripts/run_tdep.py --config tdep_tetragonal.yaml
```

`external/tdep` is pinned to a specific official revision. To intentionally update it later, use `git submodule update --remote external/tdep`, test the workflow, and commit the resulting gitlink change.

Set `tdep.temperature_K` in `tdep_tetragonal.yaml` before production if a temperature other than 300 K is required. If a calculation stops, rerun the same command: completed iterations are reused.

## Outputs

The configured output directory (`tdep_tetragonal_300K/` by default) contains per-iteration TDEP inputs/outputs, SevenNet energies and forces, fitted FC2 (`outfile.forceconstant`), and phonon bands. `phonon_dispersion_by_iteration.png` overlays each iteration so renormalization convergence is visible; `convergence.csv` records the dispersion change.

These generated outputs are intentionally ignored by Git. Keep an archived result directory or a DOI-backed data repository for production data that should be shared.
