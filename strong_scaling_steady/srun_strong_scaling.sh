#!/usr/bin/env bash
set -euo pipefail

#to not evenly spread across nodes to show bottleneck on strong scaling
MPI_LAUNCHER="srun --distribution=block:block,Pack"

# Refuse running on login node
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: Run this via salloc/sbatch (not on login node)."
  exit 1
fi


EXE_MF=./poisson_mf_strong
EXE_MAT=./poisson_mat_strong

N=80
PROCS_LIST=(1 2 4 8 16 32 64 72 96 128 144)

OUTDIR=strong_scaling_results
mkdir -p "${OUTDIR}"

echo "Running strong scaling for N=${N}"
echo "Results will be in ${OUTDIR}/"

for P in "${PROCS_LIST[@]}"; do
  echo ">>> MATRIX-FREE with ${P} MPI processes"
  ${MPI_LAUNCHER} -n "${P}" "${EXE_MF}" "${N}" "strong_scaling_mf_tmp.csv"
  mv strong_scaling_mf_tmp.csv "${OUTDIR}/mf_np${P}.csv"

  echo ">>> MATRIX-BASED with ${P} MPI processes"
  ${MPI_LAUNCHER} -n "${P}" "${EXE_MAT}" "${N}" "strong_scaling_mat_tmp.csv"
  mv strong_scaling_mat_tmp.csv "${OUTDIR}/mat_np${P}.csv"
done

echo "All runs completed."

