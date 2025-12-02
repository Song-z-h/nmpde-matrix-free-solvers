import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------
# Load data
# ----------------------------------------------------
df = pd.read_csv("scaling_all.csv")

# Separate MF and sparse
df_mf     = df[df["backend"] == "mf"].copy()
df_sparse = df[df["backend"] == "sparse"].copy()

# Sort by mpi_size just in case
df_mf.sort_values("mpi_size", inplace=True)
df_sparse.sort_values("mpi_size", inplace=True)

# ----------------------------------------------------
# Strong scaling: total_linear_solve_time vs mpi_size
# ----------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "lines.linewidth": 2,
    "lines.markersize": 7,
})

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ax_time, ax_speedup, ax_dofs = axes

# 1) Linear solve time vs mpi size
ax_time.plot(df_mf["mpi_size"], df_mf["total_linear_solve_time"],
             marker="o", label="MF")
ax_time.plot(df_sparse["mpi_size"], df_sparse["total_linear_solve_time"],
             marker="s", label="Sparse")

ax_time.set_xlabel("MPI ranks")
ax_time.set_ylabel("Total linear solve time [s]")
ax_time.set_title("Strong scaling: linear solve time")
ax_time.grid(True, which="both", alpha=0.4)
ax_time.set_xticks(df_mf["mpi_size"])
ax_time.legend()

# 2) Speedup vs mpi size (relative to 1 rank)
def compute_speedup(df_backend):
    df_backend = df_backend.copy()
    t1 = df_backend.loc[df_backend["mpi_size"] == 1, "total_linear_solve_time"]
    if len(t1) == 0:
        # If you didn't run with 1 core, skip speedup
        return df_backend["mpi_size"].values, None
    t1 = float(t1.iloc[0])
    speedup = t1 / df_backend["total_linear_solve_time"]
    return df_backend["mpi_size"].values, speedup.values

p_mf, speed_mf = compute_speedup(df_mf)
p_sp, speed_sp = compute_speedup(df_sparse)

if speed_mf is not None:
    ax_speedup.plot(p_mf, speed_mf, marker="o", label="MF")
if speed_sp is not None:
    ax_speedup.plot(p_sp, speed_sp, marker="s", label="Sparse")

# Ideal speedup line
if len(df_mf) > 0:
    p = df_mf["mpi_size"].values
    ax_speedup.plot(p, p / p[0], "--", color="gray", label="Ideal")

ax_speedup.set_xlabel("MPI ranks")
ax_speedup.set_ylabel("Speedup (T1 / Tp)")
ax_speedup.set_title("Strong scaling: speedup")
ax_speedup.grid(True, which="both", alpha=0.4)
ax_speedup.set_xticks(df_mf["mpi_size"])
ax_speedup.legend()

# 3) DoFs per second vs mpi size
ax_dofs.plot(df_mf["mpi_size"], df_mf["dofs_per_second"],
             marker="o", label="MF")
ax_dofs.plot(df_sparse["mpi_size"], df_sparse["dofs_per_second"],
             marker="s", label="Sparse")

ax_dofs.set_xlabel("MPI ranks")
ax_dofs.set_ylabel("DoFs / second")
ax_dofs.set_title("Strong scaling: throughput")
ax_dofs.grid(True, which="both", alpha=0.4)
ax_dofs.set_xticks(df_mf["mpi_size"])
ax_dofs.legend()

plt.tight_layout()
plt.savefig("strong_scaling_mf_vs_sparse.png", dpi=300, bbox_inches="tight")
plt.show()
