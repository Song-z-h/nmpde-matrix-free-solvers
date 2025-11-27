#include "Poisson3D_parallel_mf.hpp"

void Poisson3DParallelMf::setup()
{
  pcout << "===============================================" << std::endl;
  pcout << "Initializing the mesh (Hypercube Generator)" << std::endl;

  GridGenerator::subdivided_hyper_cube(mesh, N, 0.0, 1.0);
  mesh.refine_global(1);

  pcout << "  Subdivisions per axis: " << N << std::endl;
  pcout << "  Number of elements = " << mesh.n_global_active_cells() << std::endl;
  pcout << "-----------------------------------------------" << std::endl;

  // 3. Finite Element Space
  {
    fe = std::make_unique<FE_Q<dim>>(r);
    quadrature = std::make_unique<QGauss<dim>>(r + 1);

    pcout << "  Degree = " << fe->degree << std::endl;
    pcout << "  DoFs per cell = " << fe->dofs_per_cell << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;
    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize constraints (Dirichlet + hanging nodes)
  {
    pcout << "Initializing the linear system" << std::endl;
    pcout << "Building the constraints" << std::endl;
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    for (const auto &id : mesh.get_boundary_ids())
      boundary_functions[id] = &function_g;

    MappingFE<dim> mapping(*fe);
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             boundary_functions,
                                             constraints);
    constraints.close();
  }

  // --- Global MatrixFree + operator (one level) ---
  {
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim, Number>::AdditionalData::none;
    additional_data.mapping_update_flags =
      (update_gradients | update_values | update_JxW_values | update_quadrature_points);
    // additional_data.mg_level left at default: global operator

    MappingQ1<dim> mapping_global;

    mf_storage = std::make_shared<MatrixFree<dim, Number>>();
    mf_storage->reinit(mapping_global,
                       dof_handler,
                       constraints,
                       QGauss<1>(fe->degree + 1),
                       additional_data);

    pcout << "  Initializing the system right-hand side" << std::endl;
    mf_storage->initialize_dof_vector(system_rhs);
    pcout << "  Initializing the solution vector" << std::endl;
    mf_storage->initialize_dof_vector(solution);

    pcout << "  Initializing operator..." << std::endl;
    mf_operator.initialize(mf_storage);

    pcout << "  Evaluating coefficients..." << std::endl;
    mf_operator.evaluate_coefficient(diffusion_coefficient, reaction_coefficient);
  }

  // --- Geometric Multigrid setup (matrix-free on all levels) ---
  {
    pcout << "  Setting up Multigrid..." << std::endl;

    // A. MG DoFs + constraints
    dof_handler.distribute_mg_dofs();

    mg_constrained_dofs.clear();
    mg_constrained_dofs.initialize(dof_handler);

    std::set<types::boundary_id> boundary_ids;
    for (const auto &id : mesh.get_boundary_ids())
      boundary_ids.insert(id);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, boundary_ids);

    // B. Level containers
    const unsigned int n_levels = dof_handler.get_triangulation().n_global_levels();

    mg_mf_storage.resize(0, n_levels - 1);
    mg_matrices.resize(0, n_levels - 1);
    mg_interface_matrices.resize(0, n_levels - 1);

    MGLevelObject<typename SmootherType::AdditionalData> smoother_data_container;
    smoother_data_container.resize(0, n_levels - 1);

    // C. Build per-level MatrixFree + operators
    MappingQ1<dim> mapping_mg;
    const QGauss<1> level_quadrature(fe->degree + 1);

    for (unsigned int level = 0; level < n_levels; ++level)
    {
      // MatrixFree AdditionalData for this level
      typename MatrixFree<dim, Number>::AdditionalData level_data;
      level_data.tasks_parallel_scheme =
        MatrixFree<dim, Number>::AdditionalData::none;
      level_data.mapping_update_flags =
        (update_gradients | update_values | update_JxW_values |
         update_quadrature_points);
      level_data.mg_level = level;

      // Level constraints: use what MGConstrainedDoFs built for us
      const AffineConstraints<double> &level_constraints =
        mg_constrained_dofs.get_level_constraints(level);

      // Create and init MatrixFree object on this level
      std::shared_ptr<MatrixFree<dim, Number>> mf_level(
        new MatrixFree<dim, Number>());
      mf_level->reinit(mapping_mg,
                       dof_handler,
                       level_constraints,
                       level_quadrature,
                       level_data);

      mg_mf_storage[level] = mf_level;

      // Level operator (Laplace)
      mg_matrices[level].clear();
      mg_matrices[level].initialize(mf_level, mg_constrained_dofs, level);

      // Level-dependent coefficients (here just constant)
      DiffusionCoefficient<dim> diff_lvl;
      ReactionCoefficient<dim>  react_lvl;
      mg_matrices[level].evaluate_coefficient(diff_lvl, react_lvl);

      // Diagonal for Chebyshev smoother
      mg_matrices[level].compute_diagonal();

      // Smoother parameters (Chebyshev)
      typename SmootherType::AdditionalData data;
      data.smoothing_range       = 15.0;
      data.degree                = 5;
      data.eig_cg_n_iterations   = 10;
      data.preconditioner        = mg_matrices[level].get_matrix_diagonal_inverse();

      smoother_data_container[level] = data;

      // Interface operator for this level (used for edge matrices)
      mg_interface_matrices[level].initialize(mg_matrices[level]);
    }

    // D. Smoother wrapper
    mg_smoother.initialize(mg_matrices, smoother_data_container);
    mg_smoother.set_steps(2);

    // E. Matrix wrappers (volume + interface operators)
    mg_matrix_wrapper.initialize(mg_matrices);
    mg_interface_wrapper.initialize(mg_interface_matrices);

    // (No MGTransferMatrixFree here; we build it in solve() so we can rebuild
    //  easily on each new mesh.)
  }

  // Simple memory report
  const double memory_mb = get_memory_consumption();
  pcout << "  > Precise Memory (MF + Vecs): " << std::fixed << std::setprecision(4)
        << memory_mb << " MB" << std::endl;
}


void Poisson3DParallelMf::assemble()
{
  pcout << "===============================================" << std::endl;
  pcout << "Assembling RHS (Matrix-Free Style)" << std::endl;

  // Zero the RHS vector
  system_rhs = 0.0;

  // Use FEEvaluation to integrate the forcing term
  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(*mf_storage);

  for (unsigned int cell = 0; cell < mf_storage->n_cell_batches(); ++cell)
  {
    phi.reinit(cell);

    for (const unsigned int q : phi.quadrature_point_indices())
    {
      const auto p = phi.quadrature_point(q);   // Point<dim, VectorizedArray<Number>>
      const auto f_val = forcing_term.value(p); // vectorized value()
      phi.submit_value(f_val, q);
    }

    phi.integrate(EvaluationFlags::values);
    phi.distribute_local_to_global(system_rhs);
  }

  // Compress the result
  system_rhs.compress(VectorOperation::add);
}

void Poisson3DParallelMf::solve()
{
  pcout << "===============================================" << std::endl;

  // Apply Dirichlet constraints to RHS
  constraints.set_zero(system_rhs);

  // Global vectors for MF operator
  VectorType x, b;
  mf_storage->initialize_dof_vector(x);
  mf_storage->initialize_dof_vector(b);

  x = 0.0;
  b = system_rhs;
  b.update_ghost_values();

  // Choose which preconditioner to test:
  const bool use_gmg = true;   // <-- flip this to false for Identity baseline

  SolverControl solver_control(1000, 1e-8 * b.l2_norm());
  SolverCG<VectorType> solver(solver_control);

  if (!use_gmg)
  {
    pcout << "Solving with global matrix-free operator (Identity preconditioner)" << std::endl;
    PreconditionIdentity preconditioner;
    solver.solve(mf_operator, x, b, preconditioner);
  }
  else
  {
    pcout << "Solving with global matrix-free operator (GMG preconditioner)" << std::endl;

    // --- GMG preconditioner path (your current code) ---
    MGTransferMatrixFree<dim, Number> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(dof_handler);

    SolverControl coarse_control(1000, 1e-12, false, false);
    SolverCG<VectorType> coarse_solver(coarse_control);
    PreconditionIdentity identity;

    MGCoarseGridIterativeSolver<VectorType,
                                SolverCG<VectorType>,
                                LevelMatrixType,
                                PreconditionIdentity>
      coarse_grid(coarse_solver, mg_matrices[0], identity);

    mg::Matrix<VectorType> mg_m(mg_matrices);

    Multigrid<VectorType> mg(mg_m,
                             coarse_grid,
                             mg_transfer,
                             mg_smoother,
                             mg_smoother);

    PreconditionMG<dim, VectorType, MGTransferMatrixFree<dim, Number>>
      preconditioner(dof_handler, mg, mg_transfer);

    solver.solve(mf_operator, x, b, preconditioner);
  }

  constraints.distribute(x);
  solution = x;

  pcout << "  CG iterations: " << solver_control.last_step() << std::endl;
}



void Poisson3DParallelMf::output() const
{
  pcout << "===============================================" << std::endl;

  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  // To correctly export the solution, each process needs to know the solution
  // DoFs it owns, and the ones corresponding to elements adjacent to the ones
  // it owns (the locally relevant DoFs, or ghosts). We create a vector to store
  // them.
  VectorType solution_ghost(locally_owned_dofs,
                            locally_relevant_dofs,
                            MPI_COMM_WORLD);

  solution_ghost = solution;
  solution_ghost.update_ghost_values(); // FIXED: Update ghosts
  // This performs the necessary communication so that the locally relevant DoFs
  // are received from other processes and stored inside solution_ghost.

  // Then, we build and fill the DataOut class as usual.
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution_ghost, "solution");

  // We also add a vector to represent the parallel partitioning of the mesh.
  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  // Assuming N is an int or unsigned int
  const std::filesystem::path mesh_path("mesh-cube-" + std::to_string(N) + ".msh");
  const std::string output_file_name = "output-" + mesh_path.stem().string();

  // Finally, we need to write in a format that supports parallel output. This
  // can be achieved in multiple ways (e.g. XDMF/H5). We choose VTU/PVTU files,
  // because the interface is nice and it is quite robust.
  data_out.write_vtu_with_pvtu_record("./",
                                      output_file_name,
                                      0,
                                      MPI_COMM_WORLD);

  pcout << "Output written to " << output_file_name << std::endl;

  pcout << "===============================================" << std::endl;
}

double
Poisson3DParallelMf::compute_error(const VectorTools::NormType &norm_type) const
{
  FE_Q<dim> fe_linear(1);
  MappingFE mapping(fe_linear);
  // The error is an integral, and we approximate that integral using a
  // quadrature formula. To make sure we are accurate enough, we use a
  // quadrature formula with one node more than what we used in assembly.
  const QGauss<dim> quadrature_error(r + 2);

  VectorType ghosted_solution(locally_owned_dofs,
                              locally_relevant_dofs,
                              MPI_COMM_WORLD);
  ghosted_solution = solution;
  ghosted_solution.update_ghost_values(); // FIXED: Update ghosts
  // First we compute the norm on each element, and store it in a vector.
  Vector<double> error_per_cell(mesh.n_active_cells());
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    ghosted_solution,
                                    ExactSolution(),
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  // Then, we add out all the cells.
  const double error =
      VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}

double Poisson3DParallelMf::get_memory_consumption() const
{
  // 1. Matrix-Free Storage (replaces system_matrix)
  double memory_matrix = mf_storage->memory_consumption();

  // 2. Vectors
  double memory_vectors = system_rhs.memory_consumption() + solution.memory_consumption();

  // 3. Grid/DoF
  double memory_grid = mesh.memory_consumption() + dof_handler.memory_consumption();

  double local_memory = memory_matrix + memory_vectors + memory_grid;
  double global_memory = Utilities::MPI::sum(local_memory, MPI_COMM_WORLD);

  return global_memory / 1024.0 / 1024.0; // MB
}