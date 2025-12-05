import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import shutil
import numpy as np

# ================= CONFIGURATION =================

# 1. PATHS TO YOUR EXECUTABLES (Relative to this script's location)
bin_mb = "./lab-05/build/lab-05_parallel"
bin_mf = "./lab-05_mf/build/lab-05_parallel"

# 2. SCALING SETTINGS
core_counts = [1, 2, 4, 6] 
target_h = 0.025  # Corresponds to Mesh 20 (1/20)

# =================================================

results = []

print("Starting benchmark. Cleaning old data...")
# Clean up old CSVs in the root folder
for f in glob.glob("results_*.csv"):
    os.remove(f)

print(f"{'Method':<15} | {'Cores':<5} | {'Status':<10}")
print("-" * 35)

methods = [
    {"name": "Matrix-Based", "bin": bin_mb, "color": "black", "marker": "o"},
    {"name": "Matrix-Free",  "bin": bin_mf, "color": "blue",  "marker": "D"}
]

for method in methods:
    if not os.path.isfile(method['bin']): 
        print(f"ERROR: Executable not found at {method['bin']}")
        continue

    for cores in core_counts:
        # Define the directory where the binary runs and produces output
        work_dir = os.path.dirname(method['bin']) # e.g., './lab-05/build'
        
        # Define the hardcoded input/output file paths relative to the work_dir
        temp_csv_name = "convergence.csv" # The file name C++ hardcodes
        temp_csv_path = os.path.join(work_dir, temp_csv_name)
        
        # The unique name the Python script will use
        final_csv_name = f"results_{method['name'].lower().replace('-','_')}_{cores}.csv"

        # 1. Clean up old temporary CSV before running
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)

        # 2. Command: Run the executable from its build folder (work_dir)
        # We assume the C++ code uses NO ARGS and outputs "convergence.csv" locally.
        cmd_local = ["mpirun", "--allow-run-as-root", "-np", str(cores), f"./{os.path.basename(method['bin'])}"]
        
        try:
            subprocess.run(
                cmd_local, 
                cwd=work_dir, # Run executable FROM its build directory (fixes mesh path issue)
                check=True, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                env=os.environ.copy() 
            )
            
            # 3. Check if C++ finished and produced the temporary CSV
            if os.path.exists(temp_csv_path):
                # 4. Move the CSV from the build folder to the root folder, renaming it
                shutil.move(temp_csv_path, final_csv_name)
                
                # 5. Read the newly moved CSV from the current (root) directory
                df = pd.read_csv(final_csv_name) 
                
                # 6. Filter for the specific mesh size (h)
                row = df[abs(df['h'] - target_h) < 1e-6]
                
                if not row.empty:
                    vals = row.iloc[0]
                    results.append({
                        "method": method['name'],
                        "cores": cores,
                        "solve_time": vals['solve_time'],
                        "assemble_time": vals['assemble_time'],
                        "total_time": vals['solve_time'] + vals['assemble_time'],
                        "memory": vals['memory_MB'] if 'memory_MB' in df.columns else 0
                    })
                    print(f"{method['name']:<15} | {cores:<5} | Done")
                else:
                    print(f"{method['name']:<15} | {cores:<5} | Data for h={target_h} not found")
            else:
                 print(f"{method['name']:<15} | {cores:<5} | CRASHED (No CSV found)")

        except subprocess.CalledProcessError:
            print(f"{method['name']:<15} | {cores:<5} | CRASHED (CalledProcessError)")
        except Exception as e:
            print(f"{method['name']:<15} | {cores:<5} | Error reading data: {e}")

# ================= PLOTTING =================
# --- Section 1: Strong Scaling (Time vs Cores) ---
if not results:
    print("No results collected for plotting.")
    exit()

df_res = pd.DataFrame(results)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for method in methods:
    subset = df_res[df_res['method'] == method['name']]
    if not subset.empty:
        # Solve Time
        ax1.loglog(subset['cores'], subset['solve_time'], 
                   label=f"{method['name']} Solve", 
                   color=method['color'], marker=method['marker'], linestyle='-')
        
        # Ideal scaling reference (dashed)
        # Handle case where the first core time might not be T1 if T1 run failed
        t0 = subset['solve_time'].min() 
        c0 = subset['cores'].min() 
        ideal = t0 * (c0 / subset['cores'])
        ax1.loglog(subset['cores'], ideal, linestyle=':', color=method['color'], alpha=0.5, label=f"{method['name']} Ideal")

ax1.set_xlabel('Number of Cores')
ax1.set_ylabel('Wall Time [s]')
ax1.set_title(f'Strong Scaling (Mesh h={target_h})')
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xticks(core_counts)
ax1.set_xticklabels(core_counts)
ax1.grid(True, which="both", alpha=0.3)
ax1.legend()

# --- Section 2: Speedup (T1 / Tn) ---
for method in methods:
    subset = df_res[df_res['method'] == method['name']]
    if not subset.empty:
        # Calculate Speedup relative to the smallest core count (T_min_cores / T_n_cores)
        t_base = subset['solve_time'].iloc[0]
        speedup = t_base / subset['solve_time']
        
        ax2.plot(subset['cores'], speedup, 
                 label=method['name'], color=method['color'], marker=method['marker'])

# Perfect linear speedup reference
ax2.plot(core_counts, [c/core_counts[0] for c in core_counts], 'k--', label='Perfect Linear')

ax2.set_xlabel('Number of Cores')
ax2.set_ylabel('Speedup ($T_{min} / T_n$)')
ax2.set_title('Parallel Efficiency')
ax2.set_xticks(core_counts)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig("scalability_comparison.png", dpi=300)
print("\nSuccess! Plot saved to 'scalability_comparison.png'")