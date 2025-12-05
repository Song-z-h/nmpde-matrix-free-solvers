#!/usr/bin/env python3
import os
import glob
import csv
from collections import defaultdict

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
      # each file should contain exactly one row
      rows = list(reader)
      if not rows:
        continue
      row = rows[0]
      p = int(row["nprocs"])
      data_by_p[p] = row

  # sort by nprocs
  ps = sorted(data_by_p.keys())
  return ps, data_by_p


def to_float(row, key):
  return float(row[key]) if row[key] != "" else 0.0


def main():
  # Load MF data
  ps_mf, mf = load_family("mf_np*.csv")
  # Load matrix-based data
  ps_mat, mat = load_family("mat_np*.csv")

  if not ps_mf and not ps_mat:
    print("No data found. Did you run run_strong_scaling.sh?")
    return

  # -----------------------------
  # Extract metrics for plotting
  # -----------------------------
  def extract_series(ps, data, time_key="solve_time"):
    times = []
    mdofs = []
    for p in ps:
      row = data[p]
      times.append(to_float(row, time_key))
      mdofs.append(to_float(row, "million_dofs_per_second"))
    return times, mdofs

  # MF
  times_mf, mdofs_mf = extract_series(ps_mf, mf, "solve_time")
  # Matrix-based
  times_mat, mdofs_mat = extract_series(ps_mat, mat, "solve_time")

  # -----------------------------
  # Compute speedup & efficiency
  # -----------------------------
  def speedup_efficiency(ps, times):
    if not ps:
      return [], [], []
    T1 = times[0]  # assume first p is smallest, e.g. p=1
    S = [T1 / t if t > 0 else 0.0 for t in times]
    E = [S_i / p for S_i, p in zip(S, ps)]
    return S, E

  S_mf, E_mf = speedup_efficiency(ps_mf, times_mf)
  S_mat, E_mat = speedup_efficiency(ps_mat, times_mat)

  # Ideal speedup for reference (based on union of all p's)
  all_ps = sorted(set(ps_mf) | set(ps_mat))
  ideal_speedup = [p for p in all_ps]

  # -----------------------------
  # PLOTS
  # -----------------------------
  plt.figure(figsize=(7, 5))
  if ps_mf:
    plt.plot(ps_mf, times_mf, "o-", label="MF solve time")
  if ps_mat:
    plt.plot(ps_mat, times_mat, "s-", label="MAT solve time")
  plt.xlabel("Number of MPI processes")
  plt.ylabel("Solve time [s]")
  plt.title("Strong scaling: solve time vs #procs")
  plt.grid(True)
  plt.legend()
  # no log scales
  plt.tight_layout()
  plt.savefig("strong_scaling_solve_time.png", dpi=200)

  # Speedup
  plt.figure(figsize=(7, 5))
  if ps_mf:
    plt.plot(ps_mf, S_mf, "o-", label="MF speedup")
  if ps_mat:
    plt.plot(ps_mat, S_mat, "s-", label="MAT speedup")
  #plt.plot(all_ps, ideal_speedup, "k--", label="Ideal speedup")
  plt.xlabel("Number of MPI processes")
  plt.ylabel("Speedup (T1 / Tp)")
  plt.title("Strong scaling: speedup")
  plt.grid(True)
  plt.legend()
  # no log scales
  plt.tight_layout()
  plt.savefig("strong_scaling_speedup.png", dpi=200)

  # Efficiency
  plt.figure(figsize=(7, 5))
  if ps_mf:
    plt.plot(ps_mf, E_mf, "o-", label="MF efficiency")
  if ps_mat:
    plt.plot(ps_mat, E_mat, "s-", label="MAT efficiency")
  plt.xlabel("Number of MPI processes")
  plt.ylabel("Parallel efficiency S(p)/p")
  plt.title("Strong scaling: efficiency")
  plt.grid(True)
  plt.legend()
  # no log scales
  plt.tight_layout()
  plt.savefig("strong_scaling_efficiency.png", dpi=200)

  # DoFs/s
  plt.figure(figsize=(7, 5))
  if ps_mf:
    plt.plot(ps_mf, mdofs_mf, "o-", label="MF million DoFs/s")
  if ps_mat:
    plt.plot(ps_mat, mdofs_mat, "s-", label="MAT million DoFs/s")
  plt.xlabel("Number of MPI processes")
  plt.ylabel("Million DoFs/s")
  plt.title("Strong scaling: throughput")
  plt.grid(True)
  plt.legend()
  # no log scales
  plt.tight_layout()
  plt.savefig("strong_scaling_throughput.png", dpi=200)

  # # -----------------------------
  # # PLOTS
  # # -----------------------------
  # plt.figure(figsize=(7, 5))
  # if ps_mf:
  #   plt.plot(ps_mf, times_mf, "o-", label="MF solve time")
  # if ps_mat:
  #   plt.plot(ps_mat, times_mat, "s-", label="MAT solve time")
  # plt.xlabel("Number of MPI processes")
  # plt.ylabel("Solve time [s]")
  # plt.title("Strong scaling: solve time vs #procs")
  # plt.grid(True)
  # plt.legend()
  # plt.xscale("log", base=2)
  # plt.yscale("log")
  # plt.tight_layout()
  # plt.savefig("strong_scaling_solve_time.png", dpi=200)

  # # Speedup
  # plt.figure(figsize=(7, 5))
  # if ps_mf:
  #   plt.plot(ps_mf, S_mf, "o-", label="MF speedup")
  # if ps_mat:
  #   plt.plot(ps_mat, S_mat, "s-", label="MAT speedup")
  # plt.plot(all_ps, ideal_speedup, "k--", label="Ideal speedup")
  # plt.xlabel("Number of MPI processes")
  # plt.ylabel("Speedup (T1 / Tp)")
  # plt.title("Strong scaling: speedup")
  # plt.grid(True)
  # plt.legend()
  # plt.xscale("log", base=2)
  # plt.tight_layout()
  # plt.savefig("strong_scaling_speedup.png", dpi=200)

  # # Efficiency
  # plt.figure(figsize=(7, 5))
  # if ps_mf:
  #   plt.plot(ps_mf, E_mf, "o-", label="MF efficiency")
  # if ps_mat:
  #   plt.plot(ps_mat, E_mat, "s-", label="MAT efficiency")
  # plt.xlabel("Number of MPI processes")
  # plt.ylabel("Parallel efficiency S(p)/p")
  # plt.title("Strong scaling: efficiency")
  # plt.grid(True)
  # plt.legend()
  # plt.xscale("log", base=2)
  # plt.tight_layout()
  # plt.savefig("strong_scaling_efficiency.png", dpi=200)

  # # DoFs/s
  # plt.figure(figsize=(7, 5))
  # if ps_mf:
  #   plt.plot(ps_mf, mdofs_mf, "o-", label="MF million DoFs/s")
  # if ps_mat:
  #   plt.plot(ps_mat, mdofs_mat, "s-", label="MAT million DoFs/s")
  # plt.xlabel("Number of MPI processes")
  # plt.ylabel("Million DoFs/s")
  # plt.title("Strong scaling: throughput")
  # plt.grid(True)
  # plt.legend()
  # plt.xscale("log", base=2)
  # plt.tight_layout()
  # plt.savefig("strong_scaling_throughput.png", dpi=200)


  print("Saved plots:")
  print("  strong_scaling_solve_time.png")
  print("  strong_scaling_speedup.png")
  print("  strong_scaling_efficiency.png")
  print("  strong_scaling_throughput.png")


if __name__ == "__main__":
  main()
