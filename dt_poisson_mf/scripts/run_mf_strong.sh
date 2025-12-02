#!/bin/bash

N=2000
DT=0.01
T=1.0
THETA=1.0

OUT="mf_strong.csv"

# Write header once
echo "backend,mpi_size,N,ndofs,T,dt,theta,setup_time,total_wall_time,total_linear_solve_time,n_time_steps,total_gmres_iters,dofs_per_second,memory_MB" > "$OUT"

for P in 1 2 4 8 16; do
  echo "Running MF strong scaling with P = $P"
  mpirun --allow-run-as-root -np $P ./heat_mf_scaling $N $DT $T $THETA >> "$OUT"
done
