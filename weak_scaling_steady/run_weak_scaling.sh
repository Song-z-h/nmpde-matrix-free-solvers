#!/usr/bin/env bash
set -e

# Base subdivisions per axis for 1 MPI rank
BASE_N=20

# List of MPI ranks to test (adapt to your cluster)
PROCS_LIST=(1 2 3 4 5 6)

MAT_EXE=./lab-05_weak_mat
MF_EXE=./lab-05_weak_mf

echo "Running weak-scaling experiments"
echo "Base N0 = ${BASE_N}"
echo "Executables: MAT=${MAT_EXE}, MF=${MF_EXE}"
echo

for P in "${PROCS_LIST[@]}"; do
  echo "============================================="
  echo "  Running with ${P} MPI processes"
  echo "============================================="

  MAT_CSV="weak_scaling_mat_p${P}.csv"
  MF_CSV="weak_scaling_mf_p${P}.csv"

  # Matrix-based
  echo "  [MAT] mpirun -np ${P} ${MAT_EXE} ${BASE_N} ${MAT_CSV}"
  mpirun  --allow-run-as-root -np  "${P}" "${MAT_EXE}" "${BASE_N}" "${MAT_CSV}"

  # Matrix-free
  echo "  [MF ] mpirun -np ${P} ${MF_EXE} ${BASE_N} ${MF_CSV}"
  mpirun --allow-run-as-root -np "${P}" "${MF_EXE}" "${BASE_N}" "${MF_CSV}"

  echo
done

echo "All weak-scaling runs completed."
echo "CSV files:"
ls -1 weak_scaling_*_p*.csv
