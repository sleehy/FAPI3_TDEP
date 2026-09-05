# FAPbI3 TDEP finite-temperature phonons

Configuration-driven stochastic TDEP phonon renormalization for tetragonal FAPbI3. The workflow uses the native TDEP executables for quantum canonical configuration generation, force-constant fitting, and phonon dispersions, with a fine-tuned SevenNet potential for energy and force evaluation. Each iteration pauses for manual screening of SevenNet MLP energies before FC2 fitting.

## Repository contents

- `tetragonal/CONTCAR_tet`: tetragonal FAPbI3 input structure.
- `orthorhombic/CONTCAR`: companion orthorhombic structure.
- `checkpoint_fine_tuned_al_round2.pth`: SevenNet model checkpoint used by the workflow.
- `external/tdep`: pinned [official TDEP](https://github.com/tdep-developers/tdep) Git submodule.
- `tdep_tetragonal.yaml`: all run parameters.
- `scripts/run_tdep.py`: self-consistent TDEP workflow.

## Default calculation

The supplied configuration runs at 300 K with the following settings:

- 3 × 3 × 2 supercell (432 atoms for the supplied 24-atom tetragonal cell)
- Quantum canonical statistics (`canonical_configuration --quantum`), including zero-point motion
- 200 stochastic configurations per TDEP iteration
- SevenNet energy and force calculations using `checkpoint_fine_tuned_al_round2.pth`
- Native TDEP `extract_forceconstants` fitting and `phonon_dispersion_relations` on the tetragonal Γ–X–M–Γ–Z–R–A–Z path
- 6.0 Å FC2 cutoff, safely below the 6.28 Å limit of the 3 × 3 × 2 supercell along its shortest direction

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

At every incomplete iteration, the script automatically screens every generated `contcar_conf*` for abnormal Pb–I geometry before the interactive review. It reports the one-based numbers of snapshots with a Pb–I contact shorter than `screening.pb_i_min_distance_A` or a reference Pb–I bond longer than `screening.pb_i_max_bond_distance_A`, and writes all measurements to `pb_i_distance_screening.csv`. The Pb–I reference bonds are identified once from that iteration's `infile.ssposcar`; close contacts are nevertheless checked against all Pb/I pairs so a new collapsed contact is detected. The default limits (2.5–3.8 Å) are configurable in `tdep_tetragonal.yaml` and are warnings only: inspect the listed snapshots, then use `n`, `r`, `c`, or `q` as appropriate. The script also labels all snapshots with SevenNet, saves `mlp_energy_histogram.png` and a descending-energy `mlp_energies.csv`, and waits for input. Inspect any high-energy snapshot named in the CSV with VESTA. For a molecular-geometry check, enter `d` followed by one or more one-based configuration numbers (for example, `d 17` or `d 17 42`). This writes `<contcar_conf*>_bond_distance_histogram.png` and CSV files containing the N–H and C–H bond lengths. The pairs are identified once from that iteration's `infile.ssposcar` using element-specific covalent-radius cutoffs, then the same atom-index pairs are measured in the selected snapshot under periodic minimum-image distances. Thus thermal displacements cannot cause a neighbouring H atom to be mistaken for a new bond.

If a snapshot is unphysical, enter `n` followed by its one-based configuration number, for example `n 17` or `n 17 42`. The script draws fresh snapshot(s) from the same TDEP ensemble, replaces only the selected files, keeps the displaced originals in `resampled_snapshots/`, then recalculates all MLP labels and recreates the histogram for another review. You can repeat this as needed. To supply a structure manually, overwrite the same `contcar_conf*` file while preserving its supercell lattice, atom count, species, and atom ordering, then enter `r`. Enter `c` only when the full set is acceptable; it then fits FC2 and advances to the next iteration. Enter `q` (or Ctrl-C) to stop safely before FC2 fitting.

The distance diagnostic is also available independently after a run or for any compatible POSCAR pair:

```bash
python scripts/plot_bond_distances.py \
  --reference tdep_tetragonal_300K_rc2_6A/iteration_01/infile.ssposcar \
  --structure tdep_tetragonal_300K_rc2_6A/iteration_01/contcar_conf17
```

When no `--reference` is supplied, the target structure itself is used to define the bond pairs. Use `--cutoff-scale` only if a nonstandard geometry needs a different covalent-radius multiplier (the default is 1.25).

## Outputs

The configured output directory (`tdep_tetragonal_300K_rc2_6A/` by default) contains per-iteration TDEP inputs/outputs, SevenNet energies and forces, fitted FC2 (`outfile.forceconstant`), and phonon bands. Each iteration directory also contains `mlp_energy_histogram.png` and `mlp_energies.csv` for the manual configuration review. TDEP additionally writes `outfile.free_energy`, containing the phonon vibrational free energy F_vib, at `tdep.temperature_K` on the `free_energy.qpoint_grid` mesh. The root output directory collects these as `phonon_free_energy_by_iteration.png` and `phonon_free_energy_by_iteration.csv` (eV/atom).

`phonon_dispersion_by_iteration.png` overlays only iterations 1, 4, 7, … by default (`dispersion.overlay_start_iteration: 1`, `dispersion.overlay_interval: 3`), while the free-energy plot includes every completed iteration. Use these two plots for manual convergence assessment. The run always completes the exact number of `tdep.iterations` requested; there is no automatic convergence stop.

The script rejects FC2 cutoffs larger than the supercell's largest safe inscribed-sphere radius. It also refuses to reuse an output directory whose saved config differs from the active config. Change `output.directory` whenever the temperature, supercell, cutoff, or checkpoint changes. Generated outputs are intentionally ignored by Git; keep an archived result directory or a DOI-backed data repository for production data that should be shared.

## Plot a TDEP band structure and PDOS

After a TDEP iteration has fitted `outfile.forceconstant`, create a combined
dispersion and element-resolved projected DOS plot with:

```bash
python scripts/plot_tdep_phonons.py \
  --result-dir tdep_tetragonal_300K_rc2_6A/iteration_05 \
  --qmesh 24 24 24 \
  --title "FAPbI3, 300 K"
```

The script runs TDEP's official `phonon_dispersion_relations --dos` to create
`outfile.dispersion_relations.hdf5` and `outfile.phonon_dos.hdf5`, then reads
those native files to write `phonon_band_pdos.png`. Use `--reuse` to redraw an
existing pair of HDF5 outputs without recalculating phonons. TDEP stores PDOS
per symmetry-unique atom site; this script sums sites with the same chemical
element (for example, `Pb_1` and `Pb_2`) before plotting.
