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
            # Clean up keys (strip spaces)
            row = {k.strip(): v for k, v in row.items()}
            
            if "nprocs" not in row:
                continue
            
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
    # 1. Extract Metrics
    # -----------------------------
    
    # Times
    solve_mf = extract_series(ps_mf, mf, "solve_time")
    solve_mat = extract_series(ps_mat, mat, "solve_time")
    
    assemble_mf = extract_series(ps_mf, mf, "assemble_time")
    assemble_mat = extract_series(ps_mat, mat, "assemble_time")
    
    # Attempt to extract total_time (if it exists in CSV), else sum components
    if ps_mf and "total_time" in mf[ps_mf[0]]:
        total_mf = extract_series(ps_mf, mf, "total_time")
        total_mat = extract_series(ps_mat, mat, "total_time")
    else:
        print("Note: 'total_time' column not found, omitting from plot.")
        total_mf = []
        total_mat = []

    # Throughput
    mdofs_mf = extract_series(ps_mf, mf, "million_dofs_per_second")
    mdofs_mat = extract_series(ps_mat, mat, "million_dofs_per_second")

    # Memory
    mem_mf = extract_series(ps_mf, mf, "memory_MB")
    mem_mat = extract_series(ps_mat, mat, "memory_MB")

    # Solver Statistics
    iters_mf = extract_series(ps_mf, mf, "gmres_iters")
    iters_mat = extract_series(ps_mat, mat, "gmres_iters")

    # Accuracy / Physics
    err_mf = extract_series(ps_mf, mf, "eL2")
    err_mat = extract_series(ps_mat, mat, "eL2")
    
    # Problem Size (Verification of strong scaling - should be constant)
    ndofs_mf = extract_series(ps_mf, mf, "ndofs")
    ndofs_mat = extract_series(ps_mat, mat, "ndofs")

    # -----------------------------
    # 2. Compute Speedup & Efficiency
    # -----------------------------
    def speedup_efficiency(ps, times):
        if not ps:
            return [], []
        T1 = times[0]  # Reference time (p=1)
        # Speedup S = T1 / Tp
        S = [T1 / t if t > 0 else 0.0 for t in times]
        # Efficiency E = S / p
        E = [S_i / p for S_i, p in zip(S, ps)]
        return S, E

    S_mf, E_mf = speedup_efficiency(ps_mf, solve_mf)
    S_mat, E_mat = speedup_efficiency(ps_mat, solve_mat)

    # -----------------------------
    # 3. Generate Plots
    # -----------------------------
    
    def save_plot(filename, title, ylabel, log_y=True):
        plt.xlabel("Number of MPI processes")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        
        # Enforce linear X scale as requested
        plt.xscale("linear")
        
        # Conditional Y scale
        if log_y:
            plt.yscale("log")
        else:
            plt.yscale("linear")
            
        plt.savefig(filename, dpi=200)
        print(f"Saved {filename}")
        plt.close()

    # --- Plot 1: Detailed Times (Log Y, Linear X) ---
    plt.figure(figsize=(8, 6))
    if ps_mf:
        plt.plot(ps_mf, solve_mf, "o-", label="MF Solve", color="tab:blue")
        plt.plot(ps_mf, assemble_mf, "o--", label="MF Assemble", color="tab:cyan")
        if total_mf:
            plt.plot(ps_mf, total_mf, "o:", label="MF Total", color="tab:blue", alpha=0.5)
    if ps_mat:
        plt.plot(ps_mat, solve_mat, "s-", label="MAT Solve", color="tab:orange")
        plt.plot(ps_mat, assemble_mat, "s--", label="MAT Assemble", color="tab:red")
        if total_mat:
            plt.plot(ps_mat, total_mat, "s:", label="MAT Total", color="tab:orange", alpha=0.5)
    save_plot("strong_scaling_times.png", "Strong Scaling: Execution Times", "Time [s]", log_y=True)

    # --- Plot 2: Speedup (Linear Y, Linear X) ---
    plt.figure(figsize=(7, 5))
    if ps_mf:
        plt.plot(ps_mf, S_mf, "o-", label="MF Speedup")
    if ps_mat:
        plt.plot(ps_mat, S_mat, "s-", label="MAT Speedup")
    
    # Ideal speedup line (Linear)
    all_ps = sorted(list(set(ps_mf) | set(ps_mat)))
    if all_ps:
        plt.plot(all_ps, all_ps, "k--", alpha=0.5, label="Ideal (Linear)")
        
    save_plot("strong_scaling_speedup.png", "Strong Scaling: Speedup (Solve Phase)", "Speedup (T1/Tp)", log_y=False)

    # --- Plot 3: Efficiency (Linear Y, Linear X) ---
    plt.figure(figsize=(7, 5))
    if ps_mf:
        plt.plot(ps_mf, E_mf, "o-", label="MF Efficiency")
    if ps_mat:
        plt.plot(ps_mat, E_mat, "s-", label="MAT Efficiency")
    plt.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Ideal")
    
    save_plot("strong_scaling_efficiency.png", "Strong Scaling: Efficiency (Solve Phase)", "Efficiency", log_y=False)

    # --- Plot 4: Throughput (Log Y, Linear X) ---
    # DoFs/s often spans orders of magnitude, so Log Y is usually best
    plt.figure(figsize=(7, 5))
    if ps_mf:
        plt.plot(ps_mf, mdofs_mf, "o-", label="MF")
    if ps_mat:
        plt.plot(ps_mat, mdofs_mat, "s-", label="MAT")
    save_plot("strong_scaling_throughput.png", "Strong Scaling: Throughput", "Million DoFs/s", log_y=True)

    # --- Plot 5: Memory Usage (Linear Y, Linear X) ---
    plt.figure(figsize=(7, 5))
    if ps_mf:
        plt.plot(ps_mf, mem_mf, "o-", label="MF Memory")
    if ps_mat:
        plt.plot(ps_mat, mem_mat, "s-", label="MAT Memory")
    save_plot("strong_scaling_memory.png", "Strong Scaling: Total Memory Usage", "Memory [MB]", log_y=False)

    # --- Plot 6: GMRES Iterations (Linear Y, Linear X) ---
    plt.figure(figsize=(7, 5))
    if ps_mf:
        plt.plot(ps_mf, iters_mf, "o-", label="MF Iterations")
    if ps_mat:
        plt.plot(ps_mat, iters_mat, "s-", label="MAT Iterations")
    save_plot("strong_scaling_iters.png", "Strong Scaling: Solver Iterations", "GMRES Iterations", log_y=False)

    # --- Plot 7: Accuracy (Log Y, Linear X) ---
    plt.figure(figsize=(7, 5))
    if ps_mf:
        plt.plot(ps_mf, err_mf, "o-", label="MF L2 Error")
    if ps_mat:
        plt.plot(ps_mat, err_mat, "s-", label="MAT L2 Error")
    save_plot("strong_scaling_error.png", "Strong Scaling: L2 Error", "L2 Norm Error", log_y=True)

    # --- Plot 8: Problem Size (Linear Y, Linear X) ---
    plt.figure(figsize=(7, 5))
    if ps_mf:
        plt.plot(ps_mf, ndofs_mf, "o-", label="MF DoFs")
    if ps_mat:
        plt.plot(ps_mat, ndofs_mat, "s-", label="MAT DoFs")
    save_plot("strong_scaling_size.png", "Strong Scaling: Problem Size", "Total DoFs", log_y=False)

if __name__ == "__main__":
    main()
