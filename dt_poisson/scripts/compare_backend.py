import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
file_mf = "convergence_jacobi.csv"          # matrix-free + Jacobi
file_sp = "convergence_sparse_jacobi.csv"   # sparse-matrix + Jacobi

try:
    df_mf = pd.read_csv(file_mf).sort_values(by='deltat', ascending=False)
    df_sp = pd.read_csv(file_sp).sort_values(by='deltat', ascending=False)
    print("Successfully loaded MF + Sparse Jacobi datasets.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    raise SystemExit

# ==========================================
# PLOTTING STYLE
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 11,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'grid.alpha': 0.5,
    'grid.linestyle': '--',
    'axes.titlesize': 13
})

# MF: solid, Sparse: dashed
style_mf = dict(linestyle='-',  marker='o', alpha=0.9)
style_sp = dict(linestyle='--', marker='D', alpha=0.9)

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
ax_err, ax_iters, ax_time, ax_dofs, ax_mem, ax_misc = axes.ravel()

def setup_axis_dt(ax, xlabel, ylabel, title):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.invert_xaxis()  # smaller Δt to the right
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)

# ==========================================
# 1) Error vs Δt (L2 and H1)
# ==========================================
setup_axis_dt(ax_err, r'Time step $\Delta t$', 'Error',
              'Accuracy: MF Jacobi vs Sparse Jacobi')

ax_err.loglog(df_mf['deltat'], df_mf['eL2'],
              color='#1f77b4', label=r'MF Jacobi $L^2$', **style_mf)
ax_err.loglog(df_sp['deltat'], df_sp['eL2'],
              color='#1f77b4', label=r'Sparse Jacobi $L^2$', **style_sp)

ax_err.loglog(df_mf['deltat'], df_mf['eH1'],
              color='#ff7f0e', label=r'MF Jacobi $H^1$', **style_mf)
ax_err.loglog(df_sp['deltat'], df_sp['eH1'],
              color='#ff7f0e', label=r'Sparse Jacobi $H^1$', **style_sp)

ax_err.legend(loc='best')

# ==========================================
# 2) Average GMRES iterations per step
# ==========================================
setup_axis_dt(ax_iters, r'$\Delta t$', 'Avg GMRES iters/step', 'Solver Effort')

ax_iters.loglog(df_mf['deltat'], df_mf['avg_gmres_iters_per_step'],
                color='#2ca02c', label='MF Jacobi', **style_mf)
ax_iters.loglog(df_sp['deltat'], df_sp['avg_gmres_iters_per_step'],
                color='#2ca02c', label='Sparse Jacobi', **style_sp)

ax_iters.legend(loc='best')

# ==========================================
# 3) Total linear solve time
# ==========================================
setup_axis_dt(ax_time, r'$\Delta t$', 'Total linear solve time [s]',
              'Runtime: Linear Solves')

ax_time.loglog(df_mf['deltat'], df_mf['total_linear_solve_time'],
               color='#d62728', label='MF Jacobi', **style_mf)
ax_time.loglog(df_sp['deltat'], df_sp['total_linear_solve_time'],
               color='#d62728', label='Sparse Jacobi', **style_sp)

ax_time.legend(loc='best')

# ==========================================
# 4) DoFs per second
# ==========================================
setup_axis_dt(ax_dofs, r'$\Delta t$', 'DoFs/s', 'Throughput')

ax_dofs.loglog(df_mf['deltat'], df_mf['dofs_per_second'],
               color='#9467bd', label='MF Jacobi', **style_mf)
ax_dofs.loglog(df_sp['deltat'], df_sp['dofs_per_second'],
               color='#9467bd', label='Sparse Jacobi', **style_sp)

ax_dofs.legend(loc='best')

# ==========================================
# 5) Memory usage
# ==========================================
setup_axis_dt(ax_mem, r'$\Delta t$', 'Memory [MB]', 'Memory Usage')

if 'memory_MB' in df_mf.columns and 'memory_MB' in df_sp.columns:
    ax_mem.loglog(df_mf['deltat'], df_mf['memory_MB'],
                  color='black', label='MF Jacobi', **style_mf)
    ax_mem.loglog(df_sp['deltat'], df_sp['memory_MB'],
                  color='blue', label='Sparse Jacobi', **style_sp)

    # annotate last point for each
    val_mf = df_mf['memory_MB'].iloc[-1]
    val_sp = df_sp['memory_MB'].iloc[-1]

    ax_mem.annotate(f"{val_mf:.2f} MB",
                    (df_mf['deltat'].iloc[-1], val_mf),
                    xytext=(5, 10), textcoords='offset points',
                    ha='left', fontweight='bold')
    ax_mem.annotate(f"{val_sp:.2f} MB",
                    (df_sp['deltat'].iloc[-1], val_sp),
                    xytext=(5, -15), textcoords='offset points',
                    ha='left', fontweight='bold', color='blue')

    ax_mem.legend(loc='best')
else:
    ax_mem.text(0.5, 0.5, "memory_MB column missing",
                ha='center', transform=ax_mem.transAxes)

# ==========================================
# 6) Setup + error computation time
# ==========================================
setup_axis_dt(ax_misc, r'$\Delta t$', 'Time [s]', 'Overheads (Setup & Error)')

ax_misc.loglog(df_mf['deltat'], df_mf['setup_time'],
               color='#8c564b', label='MF Jacobi setup', **style_mf)
ax_misc.loglog(df_sp['deltat'], df_sp['setup_time'],
               color='#8c564b', label='Sparse Jacobi setup', **style_sp)

ax_misc.loglog(df_mf['deltat'], df_mf['error_time'],
               color='#17becf', label='MF Jacobi error comp.', **style_mf)
ax_misc.loglog(df_sp['deltat'], df_sp['error_time'],
               color='#17becf', label='Sparse Jacobi error comp.', **style_sp)

ax_misc.legend(loc='best')

plt.tight_layout()
plt.savefig("mf_jacobi_vs_sparse_jacobi.png", dpi=300, bbox_inches='tight')
print("Comparison plot saved to 'mf_jacobi_vs_sparse_jacobi.png'")
plt.show()
