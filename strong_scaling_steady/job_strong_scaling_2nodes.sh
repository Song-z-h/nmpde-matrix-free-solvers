#!/bin/bash -l
#SBATCH --job-name=poisson_strong_2nodes
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=72          # 72 cores per Ice Lake node -> max 144 ranks total
#SBATCH --time=02:00:00
#SBATCH --partition=multinode
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
set -e

# ----- user-configurable -----
EXE_MF=./poisson_mf_strong
EXE_MAT=./poisson_mat_strong

# Fixed problem size for strong scaling
N=80

# Strong-scaling process counts, up to fully utilizing 2 nodes (144 ranks)
PROCS_LIST=(1 2 4 8 16 32 64 72 96 128 144)

OUTDIR=strong_scaling_results
mkdir -p "${OUTDIR}"

# One MPI rank per physical core, no OpenMP
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Use srun with proper binding on Fritz (no pack)
MPI_LAUNCHER="srun --cpu-bind=cores --hint=nomultithread"

echo "Running strong scaling on Fritz up to 2 full nodes (144 ranks)"
echo "Problem size N = ${N}"
echo "Nodes allocated: ${SLURM_JOB_NODELIST}"
echo "Results will be in ${OUTDIR}/"
echo

for P in "${PROCS_LIST[@]}"; do
  echo "============================================="
  echo "  Running with ${P} MPI processes"
  echo "============================================="

  # Matrix-free
  echo ">>> [MF ] ${P} MPI processes"
  ${MPI_LAUNCHER} -n "${P}" "${EXE_MF}" "${N}" "strong_scaling_mf_tmp.csv"
  mv strong_scaling_mf_tmp.csv "${OUTDIR}/mf_np${P}.csv"

  # Matrix-based
  echo ">>> [MAT] ${P} MPI processes"
  ${MPI_LAUNCHER} -n "${P}" "${EXE_MAT}" "${N}" "strong_scaling_mat_tmp.csv"
  mv strong_scaling_mat_tmp.csv "${OUTDIR}/mat_np${P}.csv"

  echo
done

echo "All strong-scaling runs completed."
echo "CSV files in ${OUTDIR}/:"
ls -1 "${OUTDIR}"/mf_np*.csv "${OUTDIR}"/mat_np*.csv

