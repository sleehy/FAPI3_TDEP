#!/usr/bin/env bash
# Clone/update the pinned official TDEP submodule and build its executables.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tdep_dir="$root_dir/external/tdep"
threads="${TDEP_MAKE_THREADS:-4}"

if [[ "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: bash scripts/setup_tdep.sh

Environment variables:
  TDEP_PREFIX        Conda environment prefix used to create TDEP's
                     important_settings file on its first run.
  TDEP_MAKE_THREADS  Number of parallel build threads (default: 4).

First-run example:
  conda create -n fapi3-tdep -c conda-forge python=3.11 gfortran openmpi-mpifort scalapack fftw hdf5
  conda activate fapi3-tdep
  export TDEP_PREFIX="$CONDA_PREFIX"
  bash scripts/setup_tdep.sh
EOF
    exit 0
fi

if [[ ! -e "$tdep_dir/.git" ]]; then
    echo "Initializing the pinned official TDEP submodule..."
    git -C "$root_dir" submodule update --init --recursive
fi

required=(canonical_configuration extract_forceconstants generate_structure phonon_dispersion_relations)
all_built=true
for executable in "${required[@]}"; do
    [[ -x "$tdep_dir/bin/$executable" ]] || all_built=false
done
if [[ "$all_built" == true ]]; then
    echo "TDEP is already built: $tdep_dir/bin"
    exit 0
fi

settings="$tdep_dir/important_settings"
if [[ ! -f "$settings" ]]; then
    if [[ -z "${TDEP_PREFIX:-}" ]]; then
        cat >&2 <<EOF
TDEP needs an environment-specific build configuration.

For a Conda environment, install the dependencies documented by TDEP, activate
it, and rerun with:
  export TDEP_PREFIX="\$CONDA_PREFIX"
  bash scripts/setup_tdep.sh

For another compiler environment, create this file from the closest official
template and edit library/compiler paths before rerunning:
  $settings
EOF
        exit 2
    fi
    template="$tdep_dir/examples/build/important_settings.conda"
    sed "s|^PREFIX=.*|PREFIX=$TDEP_PREFIX|" "$template" > "$settings"
    echo "Created $settings from TDEP's Conda template."
fi

echo "Building pinned TDEP revision in $tdep_dir using $threads thread(s)..."
(cd "$tdep_dir" && bash build_things.sh --nthreads_make "$threads")
echo "TDEP build complete: $tdep_dir/bin"
