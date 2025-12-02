#!/usr/bin/env bash

# Problem parameters
N_BASE=600       # base N for strong scaling (fixed problem size)
DT=0.01
T=1.0
THETA=1.0

# MPI sizes to test
PROCS_LIST="1 2 3 4 5 6"

OUT_FILE="scaling_all.csv"

# Write header once
echo "backend,mpi_size,N,ndofs,T,dt,theta,setup_time,total_wall_time,total_linear_solve_time,n_time_steps,total_gmres_iters,dofs_per_second,memory_MB" > "$OUT_FILE"

for P in $PROCS_LIST; do
  echo "Running MF with $P ranks..."
  mpirun --allow-run-as-root -np $P ./heat_mf_scaling ${N_BASE} ${DT} ${T} ${THETA} \
    | grep -E '^(mf,)' >> "$OUT_FILE"

  echo "Running Sparse with $P ranks..."
  mpirun --allow-run-as-root -np $P ./sparse_time_convergence ${N_BASE} ${DT} ${T} ${THETA} \
    | grep -E '^(sparse,)' >> "$OUT_FILE"
done

echo "All results written to $OUT_FILE"
