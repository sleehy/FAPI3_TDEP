# FAPbI3 TDEP finite-temperature phonons

Configuration-driven self-consistent harmonic/TDEP-style phonon renormalization for tetragonal FAPbI3. The workflow uses a fine-tuned SevenNet potential for energy and force evaluation, Phonopy for canonical quantum displacement sampling and phonon bands, and symfc for symmetry-constrained second-order force matching.

## Repository contents

- `tetragonal/CONTCAR_tet`: tetragonal FAPbI3 input structure.
- `orthorhombic/CONTCAR`: companion orthorhombic structure.
- `checkpoint_fine_tuned_al_round1.pth`: SevenNet model checkpoint used by the workflow.
- `tdep_tetragonal.yaml`: all run parameters.
- `scripts/run_tdep.py`: self-consistent TDEP workflow.

## Default calculation

The supplied configuration runs at 300 K with the following settings:

- 3 × 3 × 2 supercell (432 atoms for the supplied 24-atom tetragonal cell)
- Quantum canonical statistics, including zero-point motion
- 200 stochastic configurations per TDEP iteration
- SevenNet energy and force calculations using `checkpoint_fine_tuned_al_round1.pth`
- Tetragonal Γ–X–M–Γ–Z–R–A–Z phonon path

## Setup and run

Use an environment containing Python 3.11+ and the packages in `requirements.txt`. The calculation was prepared with SevenNet 0.11.2, Phonopy 4.3.1, and symfc 1.7.3.

```bash
python -m pip install -r requirements.txt
python scripts/run_tdep.py --config tdep_tetragonal.yaml --dry-run
python scripts/run_tdep.py --config tdep_tetragonal.yaml
```

Set `tdep.temperature_K` in `tdep_tetragonal.yaml` before production if a temperature other than 300 K is required. If a calculation stops, rerun the same command: completed iterations are reused.

## Outputs

The configured output directory (`tdep_tetragonal_300K/` by default) contains per-iteration sampled displacements, SevenNet energies and forces, fitted FC2, and phonon bands. `phonon_dispersion_by_iteration.png` overlays each iteration so renormalization convergence is visible; `convergence.csv` records FC change, band change, and force-fit RMSE.

These generated outputs are intentionally ignored by Git. Keep an archived result directory or a DOI-backed data repository for production data that should be shared.
