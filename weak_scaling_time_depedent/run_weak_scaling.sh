#!/usr/bin/env bash

# ---------------------------------
# Weak scaling configuration
# ---------------------------------
N_LOCAL=64      # cells per MPI rank (approx, via N(P) = N_LOCAL * P)
DT=0.01
T=1.0
THETA=1.0

# MPI sizes to test
PROCS_LIST="1 2 3 4 5 6"

OUT_FILE="weak_scaling_all.csv"

# Write header once
echo "backend,mpi_size,N,ndofs,T,dt,theta,setup_time,total_wall_time,total_linear_solve_time,n_time_steps,total_gmres_iters,dofs_per_second,memory_MB" > "$OUT_FILE"

for P in $PROCS_LIST; do
  N_GLOBAL=$((N_LOCAL * P))

  echo "==============================================="
  echo "P = $P  (weak scaling)  --> N = $N_GLOBAL"
  echo "==============================================="

  # --- Matrix-free run ---
  echo "Running MF with $P ranks, N = $N_GLOBAL..."
  mpirun --allow-run-as-root -np $P ./heat_mf_scaling ${N_GLOBAL} ${DT} ${T} ${THETA} \
    | grep '^mf,' >> "$OUT_FILE"

  # --- Sparse run ---
  echo "Running Sparse with $P ranks, N = $N_GLOBAL..."
  mpirun --allow-run-as-root -np $P ./sparse_time_convergence ${N_GLOBAL} ${DT} ${T} ${THETA} \
    | grep '^sparse,' >> "$OUT_FILE"

done

echo "All weak scaling results written to $OUT_FILE"
