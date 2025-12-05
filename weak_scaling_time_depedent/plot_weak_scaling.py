import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------
# Load weak scaling data
# ----------------------------------------
df = pd.read_csv("weak_scaling_all.csv")

df_mf     = df[df["backend"] == "mf"].copy()
df_sparse = df[df["backend"] == "sparse"].copy()

df_mf.sort_values("mpi_size", inplace=True)
df_sparse.sort_values("mpi_size", inplace=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "lines.linewidth": 2,
    "lines.markersize": 7,
})

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ax_time, ax_eff, ax_dofs = axes

# ----------------------------------------
# 1) Total linear solve time vs mpi_size
# Ideal weak scaling: time ~ constant
# ----------------------------------------
ax_time.plot(df_mf["mpi_size"], df_mf["total_linear_solve_time"],
             marker="o", label="MF")
ax_time.plot(df_sparse["mpi_size"], df_sparse["total_linear_solve_time"],
             marker="s", label="Sparse")

ax_time.set_xlabel("MPI ranks")
ax_time.set_ylabel("Total linear solve time [s]")
ax_time.set_title("Weak scaling: linear solve time")
ax_time.grid(True, which="both", alpha=0.4)
ax_time.set_xticks(df_mf["mpi_size"])
ax_time.legend()

# ----------------------------------------
# 2) Weak scaling efficiency:
# efficiency(P) = T1 / TP
# Ideal: ~ 1 for all P
# ----------------------------------------
def weak_efficiency(df_backend):
    df_backend = df_backend.copy()
    base = df_backend.loc[df_backend["mpi_size"] == 1, "total_linear_solve_time"]
    if len(base) == 0:
        return df_backend["mpi_size"].values, None
    T1 = float(base.iloc[0])
    eff = T1 / df_backend["total_linear_solve_time"]
    return df_backend["mpi_size"].values, eff.values

p_mf,  eff_mf  = weak_efficiency(df_mf)
p_sp,  eff_sp  = weak_efficiency(df_sparse)

if eff_mf is not None:
    ax_eff.plot(p_mf, eff_mf, marker="o", label="MF")
if eff_sp is not None:
    ax_eff.plot(p_sp, eff_sp, marker="s", label="Sparse")

# ideal line: efficiency = 1
if len(df_mf) > 0:
    p = df_mf["mpi_size"].values
    ax_eff.plot(p, np.ones_like(p), "--", color="gray", label="Ideal")

ax_eff.set_xlabel("MPI ranks")
ax_eff.set_ylabel(r"Weak scaling efficiency  $T_1 / T_P$")
ax_eff.set_ylim(0, max(1.1, ax_eff.get_ylim()[1]))
ax_eff.set_title("Weak scaling: efficiency")
ax_eff.grid(True, which="both", alpha=0.4)
ax_eff.set_xticks(df_mf["mpi_size"])
ax_eff.legend()

# ----------------------------------------
# 3) DoFs per second vs mpi_size
# With weak scaling and fixed work per core,
# DoFs/s should grow ~ linearly with P.
# ----------------------------------------
ax_dofs.plot(df_mf["mpi_size"], df_mf["dofs_per_second"],
             marker="o", label="MF")
ax_dofs.plot(df_sparse["mpi_size"], df_sparse["dofs_per_second"],
             marker="s", label="Sparse")

ax_dofs.set_xlabel("MPI ranks")
ax_dofs.set_ylabel("DoFs / second")
ax_dofs.set_title("Weak scaling: throughput")
ax_dofs.grid(True, which="both", alpha=0.4)
ax_dofs.set_xticks(df_mf["mpi_size"])
ax_dofs.legend()

plt.tight_layout()
plt.savefig("weak_scaling_mf_vs_sparse.png", dpi=300, bbox_inches="tight")
plt.show()
