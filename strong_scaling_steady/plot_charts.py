#!/usr/bin/env python3
import os
import glob
import csv
import matplotlib.pyplot as plt

RESULTS_DIR = "strong_scaling_results"

def load_family(pattern):
  """
  Load a family of CSV files (e.g. mf_np*.csv) and return
  lists sorted by nprocs.
  """
  files = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
  if not files:
    print(f"No files found for pattern {pattern}")
    return [], {}

  data_by_p = {}
  for fname in files:
    with open(fname, "r", newline="") as f:
      reader = csv.DictReader(f)
      rows = list(reader)
      if not rows:
        continue
      row = rows[0]
      p = int(row["nprocs"])
      data_by_p[p] = row

  ps = sorted(data_by_p.keys())
  return ps, data_by_p

def to_float(row, key):
  if key in row and row[key] != "":
      return float(row[key])
  return 0.0

def extract_series(ps, data, key):
    values = []
    for p in ps:
      row = data[p]
      values.append(to_float(row, key))
    return values

def main():
  # Load MF data
  ps_mf, mf = load_family("mf_np*.csv")
  # Load matrix-based data
  ps_mat, mat = load_family("mat_np*.csv")

  if not ps_mf and not ps_mat:
    print("No data found. Did you run run_strong_scaling.sh?")
    return

  # -----------------------------
  # Extract metrics
  # -----------------------------
  # Solve Times
  solve_mf = extract_series(ps_mf, mf, "solve_time")
  solve_mat = extract_series(ps_mat, mat, "solve_time")
  
  # Assemble Times
  assemble_mf = extract_series(ps_mf, mf, "assemble_time")
  assemble_mat = extract_series(ps_mat, mat, "assemble_time")

  # DoFs/s
  mdofs_mf = extract_series(ps_mf, mf, "million_dofs_per_second")
  mdofs_mat = extract_series(ps_mat, mat, "million_dofs_per_second")

  # -----------------------------
  # Compute speedup & efficiency (Based on Solve Time)
  # -----------------------------
  def speedup_efficiency(ps, times):
    if not ps:
      return [], []
    T1 = times[0]  # assume first p is smallest
    S = [T1 / t if t > 0 else 0.0 for t in times]
    E = [S_i / p for S_i, p in zip(S, ps)]
    return S, E

  S_mf, E_mf = speedup_efficiency(ps_mf, solve_mf)
  S_mat, E_mat = speedup_efficiency(ps_mat, solve_mat)

  # -----------------------------
  # PLOTS
  # -----------------------------
  
  # 1. Times (Solve AND Assemble)
  plt.figure(figsize=(8, 6))
  if ps_mf:
    plt.plot(ps_mf, solve_mf, "o-", label="MF Solve", color="tab:blue")
    plt.plot(ps_mf, assemble_mf, "o--", label="MF Assemble", color="tab:cyan")
  if ps_mat:
    plt.plot(ps_mat, solve_mat, "s-", label="MAT Solve", color="tab:orange")
    plt.plot(ps_mat, assemble_mat, "s--", label="MAT Assemble", color="tab:red")
    
  plt.xlabel("Number of MPI processes")
  plt.ylabel("Time [s]")
  plt.title("Strong scaling: Assemble vs Solve Time")
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.yscale("log")
  plt.savefig("strong_scaling_times.png", dpi=200)
  print("Saved strong_scaling_times.png")

  # 2. Speedup (Solve)
  plt.figure(figsize=(7, 5))
  if ps_mf:
    plt.plot(ps_mf, S_mf, "o-", label="MF speedup")
  if ps_mat:
    plt.plot(ps_mat, S_mat, "s-", label="MAT speedup")
  
  # Ideal speedup line
  all_ps = sorted(list(set(ps_mf) | set(ps_mat)))
  if all_ps:
      plt.plot(all_ps, all_ps, "k--", alpha=0.5, label="Ideal")

  plt.xlabel("Number of MPI processes")
  plt.ylabel("Speedup (T1 / Tp)")
  plt.title("Strong scaling: Speedup (Solve Phase)")
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.savefig("strong_scaling_speedup.png", dpi=200)
  print("Saved strong_scaling_speedup.png")

  # 3. Efficiency (Solve)
  plt.figure(figsize=(7, 5))
  if ps_mf:
    plt.plot(ps_mf, E_mf, "o-", label="MF efficiency")
  if ps_mat:
    plt.plot(ps_mat, E_mat, "s-", label="MAT efficiency")
  plt.xlabel("Number of MPI processes")
  plt.ylabel("Parallel efficiency S(p)/p")
  plt.title("Strong scaling: Efficiency (Solve Phase)")
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.savefig("strong_scaling_efficiency.png", dpi=200)
  print("Saved strong_scaling_efficiency.png")

  # 4. DoFs/s
  plt.figure(figsize=(7, 5))
  if ps_mf:
    plt.plot(ps_mf, mdofs_mf, "o-", label="MF million DoFs/s")
  if ps_mat:
    plt.plot(ps_mat, mdofs_mat, "s-", label="MAT million DoFs/s")
  plt.xlabel("Number of MPI processes")
  plt.ylabel("Million DoFs/s")
  plt.title("Strong scaling: Throughput")
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.savefig("strong_scaling_throughput.png", dpi=200)
  print("Saved strong_scaling_throughput.png")

if __name__ == "__main__":
  main()
