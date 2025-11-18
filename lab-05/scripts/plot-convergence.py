import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the data
filename_csv = 'convergence.csv'

try:
    df = pd.read_csv(filename_csv)
    print(f"Successfully loaded data from {filename_csv}")
    print("Columns found:", df.columns.tolist())
except FileNotFoundError:
    print(f"Error: '{filename_csv}' not found. Run your C++ simulation first.")
    exit()

# Sort by h descending (Coarse -> Fine)
df = df.sort_values(by='h', ascending=False)

# 2. Setup Plotting
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'grid.alpha': 0.5,
    'grid.linestyle': '--'
})

# Create 3 Subplots side-by-side
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))

# ==========================================
# PLOT 1: Convergence Rates
# ==========================================
ax1.loglog(df['h'], df['eL2'], 'o-', label=r'$L^2$ Error', color='#1f77b4')
ax1.loglog(df['h'], df['eH1'], 's-', label=r'$H^1$ Error', color='#ff7f0e')

# Reference Slopes
if len(df) >= 2:
    h_ref = np.array([df['h'].iloc[-2], df['h'].iloc[-1]])
    # h^3 ref
    ref_L2 = df['eL2'].iloc[-1] * (h_ref / h_ref[-1])**3
    ax1.loglog(h_ref, ref_L2, 'k--', alpha=0.7)
    ax1.text(h_ref[0], ref_L2[0]*1.2, r'$\mathcal{O}(h^3)$', ha='right', fontsize=10)
    # h^2 ref
    ref_H1 = df['eH1'].iloc[-1] * (h_ref / h_ref[-1])**2
    ax1.loglog(h_ref, ref_H1, 'k--', alpha=0.7)
    ax1.text(h_ref[0], ref_H1[0]*1.2, r'$\mathcal{O}(h^2)$', ha='right', fontsize=10)

ax1.set_xlabel(r'Mesh Size ($h$)')
ax1.set_ylabel('Error')
ax1.set_title('Convergence Rates')
ax1.invert_xaxis()
ax1.grid(True, which="both", alpha=0.3)
ax1.legend()

# ==========================================
# PLOT 2: Computational Cost
# ==========================================
ax2.loglog(df['h'], df['solve_time'], 'd-', label='Solve', color='#d62728')
ax2.loglog(df['h'], df['assemble_time'], '^-', label='Assemble', color='#2ca02c')
ax2.loglog(df['h'], df['setup_time'], 'v-', label='Setup', color='#9467bd')

ax2.set_xlabel(r'Mesh Size ($h$)')
ax2.set_ylabel('Wall Time [s]')
ax2.set_title('Computational Cost')
ax2.invert_xaxis()
ax2.grid(True, which="both", alpha=0.3)
ax2.legend()

# ==========================================
# PLOT 3: Memory Usage (NEW)
# ==========================================
# Check if column exists (in case you use an old CSV)
if 'memory_MB' in df.columns:
    # We use a semilog plot (log x, linear y) because memory is linear with N, 
    # but h is logarithmic.
    ax3.semilogx(df['h'], df['memory_MB'], 'D-', color='black', markerfacecolor='white', markeredgewidth=2)
    
    ax3.set_xlabel(r'Mesh Size ($h$)')
    ax3.set_ylabel('Memory Consumption [MB]')
    ax3.set_title('Memory Usage (Matrix + Vectors)')
    ax3.invert_xaxis()
    ax3.grid(True, which="both", alpha=0.3)
    
    # Annotate the last point with the value
    last_h = df['h'].iloc[-1]
    last_mem = df['memory_MB'].iloc[-1]
    ax3.annotate(f"{last_mem:.1f} MB", (last_h, last_mem), 
                 xytext=(0, 10), textcoords='offset points', ha='center', fontweight='bold')
else:
    ax3.text(0.5, 0.5, "Memory column not found\nin CSV file", 
             ha='center', va='center', transform=ax3.transAxes)

plt.tight_layout()
plt.savefig("full_analysis_plot.png", dpi=300)
print("Plot saved to full_analysis_plot.png")
plt.show()