#!/usr/bin/env bash
set -euo pipefail

# Adjust if your launcher is mpiexec instead of mpirun
MPI_LAUNCHER=mpirun

# Executables (matrix-free and matrix-based)
EXE_MF=./poisson_mf_strong
EXE_MAT=./poisson_mat_strong

# Problem size N (global subdivisions per axis)
N=50

# List of MPI process counts for strong scaling
PROCS_LIST=(1 2 3 4 5 6)

# Output directory for per-run CSVs
OUTDIR=strong_scaling_results
mkdir -p "${OUTDIR}"

echo "Running strong scaling for N=${N}"
echo "Results will be in ${OUTDIR}/"

for P in "${PROCS_LIST[@]}"; do
  echo "==============================================="
  echo ">>> Running MATRIX-FREE with ${P} MPI processes"
  echo "==============================================="
  ${MPI_LAUNCHER} --allow-run-as-root -np "${P}" "${EXE_MF}" "${N}" "strong_scaling_mf_tmp.csv"
  mv strong_scaling_mf_tmp.csv "${OUTDIR}/mf_np${P}.csv"

  echo "==============================================="
  echo ">>> Running MATRIX-BASED with ${P} MPI processes"
  echo "==============================================="
  ${MPI_LAUNCHER} --allow-run-as-root -np "${P}" "${EXE_MAT}" "${N}" "strong_scaling_mat_tmp.csv"
  mv strong_scaling_mat_tmp.csv "${OUTDIR}/mat_np${P}.csv"
done

echo "All runs completed."
echo "Now run:  python3 plot_strong_scaling.py"
