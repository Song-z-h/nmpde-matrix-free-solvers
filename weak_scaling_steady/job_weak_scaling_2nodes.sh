#!/bin/bash -l
#SBATCH --job-name=poisson_weak_true_2nodes
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=72          # 72 cores per Ice Lake node -> 144 ranks max
#SBATCH --time=02:00:00
#SBATCH --partition=multinode
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
set -e

# ---------- TRUE weak scaling, but N <= 80 ----------

# Process counts (up to 2 full Ice Lake nodes)
PROCS_LIST=(1 2 4 8 16 32)

# Corresponding N(P) ≈ 15 * P^(1/3) in 3D, rounded
# This keeps DoFs per rank ~ constant and never exceeds N ≈ 80.
N_LIST=(15 19 24 30 38 48)

MAT_EXE=./lab-05_weak_mat
MF_EXE=./lab-05_weak_mf

OUTDIR=weak_scaling_results_true
mkdir -p "${OUTDIR}"

# One MPI rank per physical core, no OpenMP
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Use srun with proper binding on Fritz
MPI_LAUNCHER="srun --cpu-bind=cores --hint=nomultithread"

echo "Running TRUE weak-scaling experiments on Fritz (up to 2 nodes / 144 ranks)"
echo "Nodes allocated: ${SLURM_JOB_NODELIST}"
echo "Executables: MAT=${MAT_EXE}, MF=${MF_EXE}"
echo "Results will be in ${OUTDIR}/"
echo

# sanity check: same length
if [ ${#PROCS_LIST[@]} -ne ${#N_LIST[@]} ]; then
  echo "ERROR: PROCS_LIST and N_LIST lengths differ!" >&2
  exit 1
fi

for i in "${!PROCS_LIST[@]}"; do
  P=${PROCS_LIST[$i]}
  N=${N_LIST[$i]}

  echo "============================================="
  echo "  Running with ${P} MPI processes, N = ${N}"
  echo "============================================="

  MAT_CSV="${OUTDIR}/weak_scaling_mat_p${P}.csv"
  MF_CSV="${OUTDIR}/weak_scaling_mf_p${P}.csv"

  # Matrix-based
  echo "  [MAT] ${MPI_LAUNCHER} -n ${P} ${MAT_EXE} ${N} ${MAT_CSV}"
  ${MPI_LAUNCHER} -n "${P}" "${MAT_EXE}" "${N}" "${MAT_CSV}"

  # Matrix-free
  echo "  [MF ] ${MPI_LAUNCHER} -n ${P} ${MF_EXE} ${N} ${MF_CSV}"
  ${MPI_LAUNCHER} -n "${P}" "${MF_EXE}" "${N}" "${MF_CSV}"

  echo
done

echo "All weak-scaling runs completed."
echo "CSV files in ${OUTDIR}/:"
ls -1 "${OUTDIR}"/weak_scaling_*_p*.csv
