#!/usr/bin/env python3
import glob
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_family(pattern):
  """
  Load a family of weak-scaling CSV files like weak_scaling_mat_p*.csv.

  Returns:
      procs (np.ndarray)
      df_all (pd.DataFrame) – concatenated, sorted by nprocs
  """
  files = sorted(glob.glob(pattern))
  if not files:
    raise RuntimeError(f"No files found for pattern: {pattern}")

  records = []
  for fname in files:
    # extract p from ..._pXX.csv
    m = re.search(r"_p(\d+)\.csv$", fname)
    if not m:
      continue
    p = int(m.group(1))

    df = pd.read_csv(fname)
    if df.shape[0] != 1:
      raise RuntimeError(f"Expected a single row in {fname}, got {df.shape[0]}")

    row = df.iloc[0].copy()
    row["nprocs_file"] = p
    records.append(row)

  df_all = pd.DataFrame.from_records(records)

  # Use nprocs column from file OR from CSV; both should match
  if "nprocs" in df_all.columns:
    df_all["nprocs"] = df_all["nprocs"].astype(int)
  else:
    df_all["nprocs"] = df_all["nprocs_file"].astype(int)

  df_all.sort_values("nprocs", inplace=True)

  procs = df_all["nprocs"].to_numpy()
  return procs, df_all


def compute_weak_metrics(procs, df):
  """
  Given procs and dataframe with 'solve_time', 'million_dofs_per_second',
  compute:
    - solve times array
    - weak efficiency Ew(p) = T1 / Tp
    - speedup S(p) = T1 / Tp   (just for reference)
    - throughput array (MDoFs/s)
  """
  solve = df["solve_time"].to_numpy()
  throughput = df["million_dofs_per_second"].to_numpy()

  T1 = solve[0]  # time at smallest p (assumed p=1)
  speedup = T1 / solve
  weak_eff = speedup  # Ew(p) = T1/Tp

  return solve, weak_eff, speedup, throughput


def main():
  # ----- Load data -----
  procs_mat, df_mat = load_family("weak_scaling_mat_p*.csv")
  procs_mf, df_mf   = load_family("weak_scaling_mf_p*.csv")

  # sanity check
  if not np.array_equal(procs_mat, procs_mf):
    print("Warning: MAT and MF runs have different procs sets.")
  procs = procs_mat

  solve_mat, eff_mat, spd_mat, thr_mat = compute_weak_metrics(procs_mat, df_mat)
  solve_mf,  eff_mf,  spd_mf,  thr_mf  = compute_weak_metrics(procs_mf, df_mf)

  # For the ideal speedup line
  ideal_speedup = procs.astype(float)

  # # ----- Plot 1: solve time vs #procs -----
  # plt.figure(figsize=(6, 4))
  # plt.loglog(procs, solve_mf,  marker="o", label="MF solve time")
  # plt.loglog(procs, solve_mat, marker="s", label="MAT solve time")
  # plt.xlabel("Number of MPI processes")
  # plt.ylabel("Solve time [s]")
  # plt.title("Weak scaling: solve time vs #procs")
  # plt.grid(True, which="both", ls="--", alpha=0.5)
  # plt.legend()
  # plt.tight_layout()
  # plt.savefig("weak_scaling_solve_time.png", dpi=200)

  # # ----- Plot 2: weak efficiency (T1/Tp) -----
  # plt.figure(figsize=(6, 4))
  # plt.plot(procs, eff_mf,  marker="o", label="MF weak efficiency T1/Tp")
  # plt.plot(procs, eff_mat, marker="s", label="MAT weak efficiency T1/Tp")
  # plt.xscale("log", base=2)
  # plt.xticks(procs, procs)
  # plt.ylim(0.0, 1.1 * max(eff_mf.max(), eff_mat.max()))
  # plt.xlabel("Number of MPI processes")
  # plt.ylabel("Weak efficiency T1/Tp")
  # plt.title("Weak scaling: efficiency")
  # plt.grid(True, which="both", ls="--", alpha=0.5)
  # plt.legend()
  # plt.tight_layout()
  # plt.savefig("weak_scaling_efficiency.png", dpi=200)

  # # ----- Plot 3: effective speedup (for reference) -----
  # plt.figure(figsize=(6, 4))
  # plt.plot(procs, spd_mf,  marker="o", label="MF speedup T1/Tp")
  # plt.plot(procs, spd_mat, marker="s", label="MAT speedup T1/Tp")
  # plt.plot(procs, ideal_speedup, "k--", label="Ideal speedup p")
  # plt.xscale("log", base=2)
  # plt.xticks(procs, procs)
  # plt.xlabel("Number of MPI processes")
  # plt.ylabel("Speedup T1/Tp")
  # plt.title("Weak scaling: effective speedup")
  # plt.grid(True, which="both", ls="--", alpha=0.5)
  # plt.legend()
  # plt.tight_layout()
  # plt.savefig("weak_scaling_speedup.png", dpi=200)

  # # ----- Plot 4: throughput (MDoFs/s) vs #procs -----
  # plt.figure(figsize=(6, 4))
  # plt.plot(procs, thr_mf,  marker="o", label="MF million DoFs/s")
  # plt.plot(procs, thr_mat, marker="s", label="MAT million DoFs/s")
  # plt.xscale("log", base=2)
  # plt.xticks(procs, procs)
  # plt.xlabel("Number of MPI processes")
  # plt.ylabel("Million DoFs/s")
  # plt.title("Weak scaling: throughput")
  # plt.grid(True, which="both", ls="--", alpha=0.5)
  # plt.legend()
  # plt.tight_layout()
  # plt.savefig("weak_scaling_throughput.png", dpi=200)


  # ----- Plot 1: solve time vs #procs (linear axes) -----
  plt.figure(figsize=(6, 4))
  plt.plot(procs, solve_mf,  marker="o", label="MF solve time")
  plt.plot(procs, solve_mat, marker="s", label="MAT solve time")
  plt.xlabel("Number of MPI processes")
  plt.ylabel("Solve time [s]")
  plt.title("Weak scaling: solve time vs #procs")
  plt.grid(True, ls="--", alpha=0.5)
  plt.legend()
  plt.tight_layout()
  plt.savefig("weak_scaling_solve_time.png", dpi=200)

  # ----- Plot 2: weak efficiency (T1/Tp), linear x-axis -----
  plt.figure(figsize=(6, 4))
  plt.plot(procs, eff_mf,  marker="o", label="MF weak efficiency T1/Tp")
  plt.plot(procs, eff_mat, marker="s", label="MAT weak efficiency T1/Tp")
  plt.xticks(procs, procs)
  plt.ylim(0.0, 1.1 * max(eff_mf.max(), eff_mat.max()))
  plt.xlabel("Number of MPI processes")
  plt.ylabel("Weak efficiency T1/Tp")
  plt.title("Weak scaling: efficiency")
  plt.grid(True, ls="--", alpha=0.5)
  plt.legend()
  plt.tight_layout()
  plt.savefig("weak_scaling_efficiency.png", dpi=200)

  # ----- Plot 3: effective speedup (linear axes) -----
  plt.figure(figsize=(6, 4))
  plt.plot(procs, spd_mf,  marker="o", label="MF speedup T1/Tp")
  plt.plot(procs, spd_mat, marker="s", label="MAT speedup T1/Tp")
  #plt.plot(procs, ideal_speedup, "k--", label="Ideal speedup p")
  plt.xticks(procs, procs)
  plt.xlabel("Number of MPI processes")
  plt.ylabel("Speedup T1/Tp")
  plt.title("Weak scaling: effective speedup")
  plt.grid(True, ls="--", alpha=0.5)
  plt.legend()
  plt.tight_layout()
  plt.savefig("weak_scaling_speedup.png", dpi=200)

  # ----- Plot 4: throughput (MDoFs/s) vs #procs (linear axes) -----
  plt.figure(figsize=(6, 4))
  plt.plot(procs, thr_mf,  marker="o", label="MF million DoFs/s")
  plt.plot(procs, thr_mat, marker="s", label="MAT million DoFs/s")
  plt.xticks(procs, procs)
  plt.xlabel("Number of MPI processes")
  plt.ylabel("Million DoFs/s")
  plt.title("Weak scaling: throughput")
  plt.grid(True, ls="--", alpha=0.5)
  plt.legend()
  plt.tight_layout()
  plt.savefig("weak_scaling_throughput.png", dpi=200)

  print("Saved plots:")
  print("  weak_scaling_solve_time.png")
  print("  weak_scaling_efficiency.png")
  print("  weak_scaling_speedup.png")
  print("  weak_scaling_throughput.png")


if __name__ == "__main__":
  main()
