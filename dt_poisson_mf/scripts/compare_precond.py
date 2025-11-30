import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
file_no_prec = "convergence_no_prec.csv"
file_jacobi  = "convergence_jacobi.csv"

try:
    df_no = pd.read_csv(file_no_prec).sort_values(by='deltat', ascending=False)
    df_jac = pd.read_csv(file_jacobi).sort_values(by='deltat', ascending=False)
    print("Successfully loaded both datasets.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print(f"Make sure '{file_no_prec}' and '{file_jacobi}' exist in this folder.")
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

# Solid for no preconditioner, dashed for Jacobi
style_no  = dict(linestyle='-',  marker='o', alpha=0.9)
style_jac = dict(linestyle='--', marker='D', alpha=0.9)

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
ax_err, ax_iters, ax_time, ax_dofs, ax_mem, ax_misc = axes.ravel()

# Helper to make log-log vs Δt and invert x axis
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
setup_axis_dt(ax_err, r'Time step $\Delta t$', 'Error', 'Accuracy: Jacobi vs No Preconditioner')

ax_err.loglog(df_no['deltat'], df_no['eL2'], color='#1f77b4',
              label=r'No prec $L^2$', **style_no)
ax_err.loglog(df_jac['deltat'], df_jac['eL2'], color='#1f77b4',
              label=r'Jacobi $L^2$', **style_jac)

ax_err.loglog(df_no['deltat'], df_no['eH1'], color='#ff7f0e',
              label=r'No prec $H^1$', **style_no)
ax_err.loglog(df_jac['deltat'], df_jac['eH1'], color='#ff7f0e',
              label=r'Jacobi $H^1$', **style_jac)

ax_err.legend(loc='best')

# ==========================================
# 2) Average GMRES iterations per step
# ==========================================
setup_axis_dt(ax_iters, r'$\Delta t$', 'Avg GMRES iters/step', 'Solver Effort')

ax_iters.loglog(df_no['deltat'], df_no['avg_gmres_iters_per_step'],
                color='#2ca02c', label='No prec', **style_no)
ax_iters.loglog(df_jac['deltat'], df_jac['avg_gmres_iters_per_step'],
                color='#2ca02c', label='Jacobi', **style_jac)

ax_iters.legend(loc='best')

# ==========================================
# 3) Total linear solve time
# ==========================================
setup_axis_dt(ax_time, r'$\Delta t$', 'Total linear solve time [s]', 'Runtime: Linear Solves')

ax_time.loglog(df_no['deltat'], df_no['total_linear_solve_time'],
               color='#d62728', label='No prec', **style_no)
ax_time.loglog(df_jac['deltat'], df_jac['total_linear_solve_time'],
               color='#d62728', label='Jacobi', **style_jac)

ax_time.legend(loc='best')

# ==========================================
# 4) DoFs per second
# ==========================================
setup_axis_dt(ax_dofs, r'$\Delta t$', 'DoFs/s', 'Throughput')

ax_dofs.loglog(df_no['deltat'], df_no['dofs_per_second'],
               color='#9467bd', label='No prec', **style_no)
ax_dofs.loglog(df_jac['deltat'], df_jac['dofs_per_second'],
               color='#9467bd', label='Jacobi', **style_jac)

ax_dofs.legend(loc='best')

# ==========================================
# 5) Memory usage
# ==========================================
setup_axis_dt(ax_mem, r'$\Delta t$', 'Memory [MB]', 'Memory Usage')

if 'memory_MB' in df_no.columns and 'memory_MB' in df_jac.columns:
    ax_mem.loglog(df_no['deltat'], df_no['memory_MB'],
                  color='black', label='No prec', **style_no)
    ax_mem.loglog(df_jac['deltat'], df_jac['memory_MB'],
                  color='blue', label='Jacobi', **style_jac)

    # annotate last point
    val_no  = df_no['memory_MB'].iloc[-1]
    val_jac = df_jac['memory_MB'].iloc[-1]

    ax_mem.annotate(f"{val_no:.2f} MB",
                    (df_no['deltat'].iloc[-1], val_no),
                    xytext=(5, 10), textcoords='offset points',
                    ha='left', fontweight='bold')
    ax_mem.annotate(f"{val_jac:.2f} MB",
                    (df_jac['deltat'].iloc[-1], val_jac),
                    xytext=(5, -15), textcoords='offset points',
                    ha='left', fontweight='bold', color='blue')
    ax_mem.legend(loc='best')
else:
    ax_mem.text(0.5, 0.5, "memory_MB column missing", ha='center', transform=ax_mem.transAxes)

# ==========================================
# 6) Setup + error computation time
# ==========================================
setup_axis_dt(ax_misc, r'$\Delta t$', 'Time [s]', 'Overheads (Setup & Error)')

ax_misc.loglog(df_no['deltat'], df_no['setup_time'],
               color='#8c564b', label='No prec setup', **style_no)
ax_misc.loglog(df_jac['deltat'], df_jac['setup_time'],
               color='#8c564b', label='Jacobi setup', **style_jac)

ax_misc.loglog(df_no['deltat'], df_no['error_time'],
               color='#17becf', label='No prec error comp.', **style_no)
ax_misc.loglog(df_jac['deltat'], df_jac['error_time'],
               color='#17becf', label='Jacobi error comp.', **style_jac)

ax_misc.legend(loc='best')

plt.tight_layout()
plt.savefig("jacobi_vs_no_prec.png", dpi=300, bbox_inches='tight')
print("Comparison plot saved to 'jacobi_vs_no_prec.png'")
plt.show()
