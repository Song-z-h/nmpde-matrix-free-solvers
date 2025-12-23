### Repository layout

Top-level folders:

- `common/`  
  Shared headers, utilities, and parameter handling used by several labs.

- `lab-05_adr/`  
  Matrix-based ADR solver (Deal.II + Trilinos).  
  Used for:
  - steady diffusion–reaction (set `beta0 = 0`, use CG),
  - steady ADR with advection (set `beta0 > 0`, use GMRES),
  - spatial convergence and baseline runs.

- `lab-05_mf_adr/`  
  Matrix-free ADR solver built on `MatrixFree` + `FEEvaluation`.  
  Same PDEs, meshes, and manufactured solution as `lab-05_adr`.

- `dt_poisson/`  
  Matrix-based time-dependent diffusion (Poisson in time) with an implicit
  θ-scheme (Crank–Nicolson in the project). Used for the unsteady
  diffusion–reaction baseline.

- `dt_poisson_mf/`  
  Matrix-free version of the time-dependent diffusion solver.

- `strong_scaling_steady/`  
  Job scripts, parameter files, and helpers to reproduce the steady strong-scaling
  experiments (diffusion–reaction and ADR) on Fritz.

- `weak_scaling_steady/`  
  Job scripts and parameter files for the steady weak-scaling runs.

- `strong_scaling_time_dependent/`  
  Job scripts and parameter files for the time-dependent strong-scaling runs.

- `weak_scaling_time_depedent/`  
  Job scripts and parameter files for the time-dependent weak-scaling runs.

- `configure_fritz.txt`  
  Exact module loads and `cmake` flags used on Fritz (NHR@FAU).

- `build_docker_env.txt`  
  Minimal environment / commands to build everything inside a container or
  on a different cluster.

- `.gitignore`, `CMakeLists.txt`, `README.md`  
  Standard project files.

- All C++ sources live in the respective `*/src` folders. Shared utilities live in `common/`. The top-level `src/` is legacy and can be ignored.


### Building the labs

Each lab has its own `CMakeLists.txt`; build inside the target folder, e.g.:

    cd nmpde-matrix-free-solvers/lab-05_adr   # or lab-05_mf_adr, dt_poisson, dt_poisson_mf
    mkdir -p build && cd build

    # If a Deal.II module provides DEAL_II_DIR, a plain cmake .. is enough.
    cmake -DDEAL_II_DIR=$WORK/libs/install/dealii-9.5.1 ..
    make -j 20

Main executables (names as in the paper):

  * `lab-05_adr`:
      - `lab-05_parallel`        — manufactured-solution test and basic runs
      - `poisson_mat_strong`     — steady strong-scaling baseline (matrix-based)
      - `lab-05_weak_mat`        — steady weak-scaling baseline (matrix-based)

  * `lab-05_mf_adr`:
      - matrix-free analogues of the above (same meshes / parameters)

  * `dt_poisson`:
      - `lab-06`                 — time-dependent diffusion driver
      - `sparse_time_convergence` — time-step convergence study (matrix-based)

  * `dt_poisson_mf`:
      - `lab-06`                 — time-dependent diffusion driver (matrix-free)
      - `heat_mf_scaling`        — time-dependent scaling runs (matrix-free)


#### Diffusion–reaction vs. ADR and CG vs. GMRES

The diffusion–reaction baseline (symmetric) and the ADR problems (nonsymmetric)
share the same finite element implementation:

- Set `beta0 = 0` in the parameter file to obtain the diffusion–reaction operator.
- Set `beta0 = 1` or `beta0 = 1000` for ADR with moderate or strong advection.

In the code we use:

- Conjugate gradients (CG) for diffusion–reaction (`beta0 = 0`).
- Restarted GMRES for ADR with advection (`beta0 > 0`).

For the parallel diffusion–reaction base in the paper, we simply run
`lab-05_adr` and `lab-05_mf_adr` with `beta0 = 0` and CG.
For the ADR experiments we use the same executables but set `beta0 > 0`
and switch to GMRES.


For convenience, we also provide:

- `configure_fritz.txt`: the exact module loads and `cmake` invocation used on
  the Fritz Ice Lake nodes (NHR@FAU).
- `build_docker_env.txt`: a minimal environment / build recipe for running the
  code inside a Docker/Singularity image or on a different cluster.
