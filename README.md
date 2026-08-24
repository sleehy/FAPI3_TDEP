# FAPbI3 TDEP finite-temperature phonons

Configuration-driven stochastic TDEP phonon renormalization for tetragonal FAPbI3. The workflow uses the native TDEP executables for quantum canonical configuration generation, force-constant fitting, and phonon dispersions, with a fine-tuned SevenNet potential for energy and force evaluation.

## Repository contents

- `tetragonal/CONTCAR_tet`: tetragonal FAPbI3 input structure.
- `orthorhombic/CONTCAR`: companion orthorhombic structure.
- `checkpoint_fine_tuned_al_round1.pth`: SevenNet model checkpoint used by the workflow.
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

Use an environment containing Python 3.11+ and the packages in `requirements.txt`. Set `tdep.bin_dir` to the directory containing the TDEP executables; the supplied configuration uses `/home/eoung/tdep/bin`. The calculation was prepared with SevenNet 0.11.2 and the native TDEP build in that location.

```bash
python -m pip install -r requirements.txt
python scripts/run_tdep.py --config tdep_tetragonal.yaml --dry-run
python scripts/run_tdep.py --config tdep_tetragonal.yaml
```

Set `tdep.temperature_K` in `tdep_tetragonal.yaml` before production if a temperature other than 300 K is required. If a calculation stops, rerun the same command: completed iterations are reused.

## Outputs

The configured output directory (`tdep_tetragonal_300K/` by default) contains per-iteration TDEP inputs/outputs, SevenNet energies and forces, fitted FC2 (`outfile.forceconstant`), and phonon bands. `phonon_dispersion_by_iteration.png` overlays each iteration so renormalization convergence is visible; `convergence.csv` records the dispersion change.

These generated outputs are intentionally ignored by Git. Keep an archived result directory or a DOI-backed data repository for production data that should be shared.
