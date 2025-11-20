import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
file_mb = 'convergence_matrix_based.csv'
file_mf = 'convergence_matrix_free.csv'

# Load Data
try:
    df_mb = pd.read_csv(file_mb).sort_values(by='h', ascending=False)
    df_mf = pd.read_csv(file_mf).sort_values(by='h', ascending=False)
    print("Successfully loaded both datasets.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print(f"Please ensure you have renamed your output files to '{file_mb}' and '{file_mf}'")
    exit()

# Setup Plotting
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 12,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'grid.alpha': 0.5,
    'grid.linestyle': '--',
    'axes.titlesize': 14
})

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))

# Styles for distinction
style_mb = {'linestyle': '-', 'marker': 'o', 'alpha': 0.8} # Solid for Matrix-Based
style_mf = {'linestyle': '--', 'marker': 'D', 'alpha': 0.8} # Dashed for Matrix-Free

# ==========================================
# PLOT 1: Error Comparison
# ==========================================
# Theory: These should be IDENTICAL. Plotting them validates your implementation.

# Matrix Based (Thicker line behind)
ax1.loglog(df_mb['h'], df_mb['eL2'], color='#1f77b4', lw=4, alpha=0.4, label=r'MB $L^2$ Error')
# Matrix Free (Thinner line on top)
ax1.loglog(df_mf['h'], df_mf['eL2'], color='#1f77b4', **style_mf, label=r'MF $L^2$ Error')

# Just plot H1 for one of them to avoid clutter, or both if you suspect differences
ax1.loglog(df_mf['h'], df_mf['eH1'], color='#ff7f0e', **style_mf, label=r'MF $H^1$ Error')

ax1.set_xlabel(r'Mesh Size ($h$)')
ax1.set_ylabel('Error')
ax1.set_title('Validation: Accuracy')
ax1.invert_xaxis()
ax1.grid(True, which="both", alpha=0.3)
ax1.legend()

# ==========================================
# PLOT 2: Computational Cost Comparison
# ==========================================
# We compare "Solve" time and "Total" time (Setup+Assemble+Solve)

# Solve Time
ax2.loglog(df_mb['h'], df_mb['solve_time'], color='#d62728', label='MB Solve', **style_mb)
ax2.loglog(df_mf['h'], df_mf['solve_time'], color='#d62728', label='MF Solve', **style_mf)

# Assembly Time (Matrix-Free assembly is usually near-instant, Matrix-Based is slow)
ax2.loglog(df_mb['h'], df_mb['assemble_time'], color='#2ca02c', label='MB Assemble', **style_mb)
ax2.loglog(df_mf['h'], df_mf['assemble_time'], color='#2ca02c', label='MF Assemble', **style_mf)

ax2.set_xlabel(r'Mesh Size ($h$)')
ax2.set_ylabel('Wall Time [s]')
ax2.set_title('Performance: Runtime')
ax2.invert_xaxis()
ax2.grid(True, which="both", alpha=0.3)
ax2.legend()

# ==========================================
# PLOT 3: Memory Consumption
# ==========================================
# This is the star show for Matrix-Free methods.

if 'memory_MB' in df_mb.columns and 'memory_MB' in df_mf.columns:
    ax3.loglog(df_mb['h'], df_mb['memory_MB'], color='black', label='Matrix-Based', **style_mb)
    ax3.loglog(df_mf['h'], df_mf['memory_MB'], color='blue', label='Matrix-Free', **style_mf)
    
    # Annotate the gap at the finest mesh
    val_mb = df_mb['memory_MB'].iloc[-1]
    val_mf = df_mf['memory_MB'].iloc[-1]
    
    ax3.annotate(f"{val_mb:.0f} MB", (df_mb['h'].iloc[-1], val_mb), xytext=(-10, 10), textcoords='offset points', ha='right', fontweight='bold')
    ax3.annotate(f"{val_mf:.0f} MB", (df_mf['h'].iloc[-1], val_mf), xytext=(-10, -15), textcoords='offset points', ha='right', fontweight='bold', color='blue')
    
    # Calculate savings factor
    savings = val_mb / val_mf
    ax3.text(0.5, 0.5, f"Factor: {savings:.1f}x savings", transform=ax3.transAxes, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    ax3.set_xlabel(r'Mesh Size ($h$)')
    ax3.set_ylabel('Memory [MB]')
    ax3.set_title('Resource: Memory Usage')
    ax3.invert_xaxis()
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend()
else:
    ax3.text(0.5, 0.5, "Memory column missing", ha='center')

plt.tight_layout()
plt.savefig("method_comparison.png", dpi=300, bbox_inches='tight')
print("Comparison plot saved to 'method_comparison.png'")
plt.show()