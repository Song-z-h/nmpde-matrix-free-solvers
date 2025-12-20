#include "Heat.hpp"

void Heat::setup()
{
  /* pcout << "Initializing the mesh" << std::endl;

   Triangulation<dim> serial_hypercube_tria;
   GridGenerator::subdivided_hyper_cube(serial_hypercube_tria, N, 0.0, 1.0, true);

   Triangulation<dim> serial_simplex_tria;
   if (dim == 1)
   {
     // In 1D, no conversion needed (intervals are simplices).
     serial_simplex_tria.copy_triangulation(serial_hypercube_tria);
   }
   else
   {
     // Convert to simplices for dim > 1. Use minimal divisions (2 in 2D, 6 in 3D) for coarser mesh.
     // const unsigned int n_divisions = (dim == 2 ? 2 : 6);
     GridGenerator::convert_hypercube_to_simplex_mesh(serial_hypercube_tria,
                                                      serial_simplex_tria);
   }

   pcout << "  Number of elements in serial mesh = " << serial_simplex_tria.n_active_cells() << std::endl;

   // Distribute for parallel.
   GridTools::partition_triangulation(mpi_size, serial_simplex_tria);
   const auto construction_data =
       TriangulationDescription::Utilities::create_description_from_triangulation(serial_simplex_tria, MPI_COMM_WORLD);
   mesh.create_triangulation(construction_data);

   pcout << "  Number of elements in distributed mesh = " << mesh.n_global_active_cells() << std::endl;

   // Optional: Save the generated mesh to a file for visualization (e.g., VTK).
   if (mpi_rank == 0)
   {
     const std::string mesh_file_name = "mesh-" + std::to_string(N + 1) + ".vtk";
     GridOut grid_out;
     std::ofstream grid_out_file(mesh_file_name);
     grid_out.write_vtk(serial_simplex_tria, grid_out_file);
     pcout << "  Mesh saved to " << mesh_file_name << std::endl;
   }*/

  pcout << "Initializing the mesh" << std::endl;

  Triangulation<dim> serial_tria;
  // Hypercube mesh, no conversion to simplices
  GridGenerator::subdivided_hyper_cube(serial_tria, N, 0.0, 1.0);

  pcout << "  Number of elements in serial mesh = "
        << serial_tria.n_active_cells() << std::endl;

  // Distribute for parallel.
  GridTools::partition_triangulation(mpi_size, serial_tria);
  const auto construction_data =
      TriangulationDescription::Utilities::create_description_from_triangulation(
          serial_tria, MPI_COMM_WORLD);
  mesh.create_triangulation(construction_data);

  pcout << "  Number of elements in distributed mesh = "
        << mesh.n_global_active_cells() << std::endl;

  // Optional: Save mesh to VTK
  if (mpi_rank == 0)
  {
    const std::string mesh_file_name =
        "mesh-" + std::to_string(N) + ".vtk";
    GridOut grid_out;
    std::ofstream grid_out_file(mesh_file_name);
    grid_out.write_vtk(serial_tria, grid_out_file);
    pcout << "  Mesh saved to " << mesh_file_name << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    // 2 3 D
    // fe = std::make_unique<FE_SimplexP<dim>>(r);

    // 1D
    // fe = std::make_unique<FE_Q<dim>>(r);

    /*if (dim > 1)
    {
      fe = std::make_unique<FE_SimplexP<dim>>(fe_degree);
      // Construct the quadrature formula of the appopriate degree of exactness.
      quadrature = std::make_unique<QGaussSimplex<dim>>(fe_degree + 1);
      quadrature_boundary = std::make_unique<QGaussSimplex<dim - 1>>(fe_degree + 1);
    }
    else
    {
      fe = std::make_unique<FE_Q<dim>>(fe_degree);
      quadrature = std::make_unique<QGauss<dim>>(fe_degree + 1);
      quadrature_boundary = std::make_unique<QGauss<dim - 1>>(fe_degree + 1);
    }

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;
    // 2 3 D
    // quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);
    // 1D

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;*/

    pcout << "Initializing the finite element space" << std::endl;

    // Hypercube + FE_Q, exactly like the Poisson code
    fe = std::make_unique<FE_Q<dim>>(fe_degree);
    quadrature = std::make_unique<QGauss<dim>>(fe_degree + 1);
    // For faces (Neumann), not actually used now but kept for completeness
    quadrature_boundary = std::make_unique<QGauss<dim - 1>>(fe_degree + 1);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell << std::endl;
    pcout << "  Quadrature points per cell = " << quadrature->size() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_handler);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  pcout << "Initializing boundary values" << std::endl;
  {
    Functions::ZeroFunction<dim> bc_function;
    // 3. Initialize constraints (move this to setup)
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    // for (const auto &id : mesh.get_boundary_ids())
    boundary_functions[0] = &bc_function;

    MappingFE<dim> mapping(*fe);
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             boundary_functions,
                                             constraints);
    constraints.close();
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the matrix-free system" << std::endl;

    additional_data.tasks_parallel_scheme =
        MatrixFree<dim, NUMBER>::AdditionalData::none;
    additional_data.mapping_update_flags =
        (update_gradients | update_values | update_JxW_values | update_quadrature_points);

    MappingFE<dim> mapping_local(*fe);
    mf_storage = std::make_shared<MatrixFree<dim, NUMBER>>();
    //if (dim > 1)
      ///mf_storage->reinit(mapping_local, dof_handler, constraints,
         //                QGaussSimplex<dim>(fe_degree + 1), additional_data);
    //if (dim == 1)
      mf_storage->reinit(mapping_local, dof_handler, constraints,
                         QGauss<dim>(fe_degree + 1), additional_data);

    pcout << "  Initializing the system right-hand side" << std::endl;
    mf_storage->initialize_dof_vector(system_rhs);
    pcout << "  Initializing the solution vector" << std::endl;
    mf_storage->initialize_dof_vector(solution);
    pcout << "  Initializing the solution owned vector" << std::endl;
    mf_storage->initialize_dof_vector(solution_owned);

    // Initialize our matrix-free operator

    mf_operator_lhs.initialize(mf_storage);
    // mf_operator_lhs.set_constraints(constraints);
    mf_operator_lhs.evaluate_coefficient(mu, b, k);

    mf_operator_lhs.compute_diagonal();

    mf_operator_rhs.initialize(mf_storage);
    mf_operator_rhs.evaluate_coefficient(mu, b, k);
    // mf_operator_rhs.set_constraints(constraints);
    pcout << "Setup completed" << std::endl;
  }
}

void Heat::assemble_rhs(const double &time)
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_quadrature_points |
                              update_JxW_values);

  // Since we need to compute integrals on the boundary for Neumann conditions,
  // we also need a FEValues object to compute quantities on boundary edges
  // (faces).
  FEFaceValues<dim> fe_values_boundary(*fe,
                                       *quadrature_boundary,
                                       update_values |
                                           update_quadrature_points |
                                           update_JxW_values);

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_rhs = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    fe_values.reinit(cell);

    cell_rhs = 0.0;

    for (unsigned int q = 0; q < n_q; ++q)
    {
      // We need to compute the forcing term at the current time (tn+1) and
      // at the old time (tn). deal.II Functions can be computed at a
      // specific time by calling their set_time method.

      // Compute f(tn+1)
      forcing_term.set_time(time);
      const double f_new_loc =
          forcing_term.value(fe_values.quadrature_point(q));

      // Compute f(tn)
      forcing_term.set_time(time - deltat);
      const double f_old_loc =
          forcing_term.value(fe_values.quadrature_point(q));

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        cell_rhs(i) += (theta * f_new_loc + (1.0 - theta) * f_old_loc) *
                       fe_values.shape_value(i, q) * fe_values.JxW(q);
      }
    }
    // neuman boundary conditions
    /*
     if (cell->at_boundary())
       {
         // ...we loop over its edges (referred to as faces in the deal.II
         // jargon).
         for (unsigned int face_number = 0; face_number < cell->n_faces();
              ++face_number)
           {
             // If current face lies on the boundary, and its boundary ID (or
             // tag) is that of one of the Neumann boundaries, we assemble the
             // boundary integral.
             if (cell->face(face_number)->at_boundary() &&
                 (cell->face(face_number)->boundary_id() == 1))
               {
                 fe_values_boundary.reinit(cell, face_number);

                 for (unsigned int q = 0; q < quadrature_boundary->size(); ++q)
                   for (unsigned int i = 0; i < dofs_per_cell; ++i)
                     cell_rhs(i) +=
                       function_h.value(
                         fe_values_boundary.quadrature_point(q)) * // h(xq)
                       fe_values_boundary.shape_value(i, q) *      // v(xq)
                       fe_values_boundary.JxW(q);                  // Jq wq
               }
           }
       } */

    cell->get_dof_indices(dof_indices);
    // system_rhs.add(dof_indices, cell_rhs);

    constraints.distribute_local_to_global(cell_rhs,
                                           dof_indices, system_rhs);
  }

  system_rhs.compress(VectorOperation::add);

  // Add the term that comes from the old solution.
  // rhs_matrix.vmult_add(system_rhs, solution_owned);
  mf_operator_rhs.vmult_add(system_rhs, solution_owned);
  /*for (const auto &entry : boundary_values)
    if (system_rhs.in_local_range(entry.first))
      system_rhs(entry.first) = entry.second;*/
}

void Heat::solve_time_step()
{
  SolverControl solver_control(500000, 1e-9 * system_rhs.l2_norm());
  // SolverGMRES<VectorType> solver(solver_control);
  SolverGMRES<VectorType> solver(solver_control);

  // Time the linear solve (global wall time)
  Timer linear_timer(MPI_COMM_WORLD);
  linear_timer.restart();

  if (preconditioner_type == PreconditionerType::Jacobi)
  {
    const auto diag_ptr = mf_operator_lhs.get_matrix_diagonal_inverse();
    Assert(diag_ptr, ExcMessage("Diagonal not initialized. Did you call compute_diagonal()?"));
    solver.solve(mf_operator_lhs, solution_owned, system_rhs, *diag_ptr);
  }
  else // PreconditionerType::None
  {
    PreconditionIdentity identity;
    solver.solve(mf_operator_lhs, solution_owned, system_rhs, identity);
  }

  linear_timer.stop();
  const double this_solve_time = linear_timer.wall_time(); // seconds

  constraints.distribute(solution_owned);
  solution = solution_owned;

  const unsigned int iters = solver_control.last_step();

  // Accumulate performance counters
  total_linear_solve_time += this_solve_time;
  total_gmres_iterations += iters;

  //pcout << "  " << iters << " GMRES iterations " << std::endl;
  //pcout << "  " << solver_control.last_value() << " GMRES residual " << std::endl;
}

void Heat::output(const unsigned int &time_step) const
{

  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
  VectorType solution_ghost(locally_owned_dofs,
                            locally_relevant_dofs,
                            MPI_COMM_WORLD);
  solution_ghost = solution;
  solution_ghost.update_ghost_values(); 

  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution_ghost, "u");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record(
      "./", "output", time_step, MPI_COMM_WORLD, 3);
}

void Heat::solve()
{
  // assemble_matrices();

  pcout << "===============================================" << std::endl;

  // Reset performance counters for this run
  n_time_steps = 0;
  total_gmres_iterations = 0;
  total_linear_solve_time = 0.0;

  // Apply the initial condition.
  {
    pcout << "Applying the initial condition" << std::endl;

    VectorTools::interpolate(dof_handler, u_0, solution_owned);
    solution = solution_owned;

    // Output the initial solution.
    if (timer)
    {
      TimerOutput::Scope t(*timer, "Output");
      output(0);
    }
    else
    {
      output(0);
    }
    pcout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;
  time = 0.0;
  while (time < T)
  {
    ++time_step;
    time = time_step * deltat;
    if (time > T)
      break;
    beta.set_time(time);
    alpha.set_time(time);

    // pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
    //     << time << ":" << std::flush;

    // --- Assemble RHS ---
    if (timer)
    {
      TimerOutput::Scope t(*timer, "Assemble RHS");
      assemble_rhs(time);
    }
    else
    {
      assemble_rhs(time);
    }
    // --- Linear solve ---
    if (timer)
    {
      TimerOutput::Scope t(*timer, "Linear solve");
      solve_time_step();
    }
    else
    {
      solve_time_step();
    }

    ++n_time_steps;

    /*if (timer)
    {
      TimerOutput::Scope t(*timer, "Output");
      output(time_step);
    }
    else
    {
      output(time_step);
    }*/
  }
}

/*
double
Heat::compute_error(const VectorTools::NormType &norm_type)
{
  // The error is an integral, and we approximate that integral using a
  // quadrature formula. To make sure we are accurate enough, we use a
  // quadrature formula with one node more than what we used in assembly.

  FE_SimplexP<dim> fe_linear(1);
  MappingFE mapping(fe_linear);

  exact_solution.set_time(time);
  // First we compute the norm on each element, and store it in a vector.
  Vector<double> error_per_cell;

  VectorType ghosted_solution(locally_owned_dofs,
                              locally_relevant_dofs,
                              MPI_COMM_WORLD);
  ghosted_solution = solution;
  ghosted_solution.update_ghost_values();

  // Use a unique_ptr to hold the correct quadrature rule.
  std::unique_ptr<Quadrature<dim>> quadrature_error;

  if (dim > 1)
  {
    // For 2D/3D simplex meshes, use QGaussSimplex.
    quadrature_error = std::make_unique<QGaussSimplex<dim>>(fe_degree + 3);
  }
  else
  {
    // For 1D, use QGauss.
    quadrature_error = std::make_unique<QGauss<dim>>(fe_degree + 3);
  }

  exact_solution.set_time(time);
  // For simplex meshes (dim > 1), you need to provide a mapping.
  if (dim > 1)
  {
    // A simple linear mapping is usually sufficient for error computation.
    MappingFE<dim> mapping(FE_SimplexP<dim>(1));
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      ghosted_solution,
                                      exact_solution,
                                      error_per_cell,
                                      *quadrature_error, // Dereference the pointer
                                      norm_type);
  }
  else
  {
    VectorTools::integrate_difference(dof_handler,
                                      ghosted_solution,
                                      exact_solution,
                                      error_per_cell,
                                      *quadrature_error, // Dereference the pointer
                                      norm_type);
  }
  // Then, we add out all the cells.
  const double error =
      VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}*/

double Heat::compute_error(const VectorTools::NormType &norm_type)
{
  // Set exact solution at current time
  exact_solution.set_time(time);

  // Quadrature: one order higher than assembly
  const QGauss<dim> quadrature_error(fe_degree + 2);

  // One entry per active cell
  Vector<double> error_per_cell(mesh.n_active_cells());

  // Ghosted solution for integration
  VectorType ghosted_solution(locally_owned_dofs,
                              locally_relevant_dofs,
                              MPI_COMM_WORLD);
  ghosted_solution = solution;
  // ghosted_solution.update_ghost_values(); // optional with Trilinos

  // Use a simple mapping (linear) for error computation
  FE_Q<dim> fe_linear(1);
  MappingFE<dim> mapping(fe_linear);

  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    ghosted_solution,
                                    exact_solution,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  const double error =
      VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}

double Heat::get_memory_consumption() const
{
  // 1. MatrixFree storage
  double memory_mf = 0.0;
  if (mf_storage)
    memory_mf = mf_storage->memory_consumption();

  // 2. Vectors (system_rhs, solution, solution_owned)
  double memory_vectors = system_rhs.memory_consumption() + solution.memory_consumption() + solution_owned.memory_consumption();

  // 3. Grid + DoFHandler
  double memory_grid = mesh.memory_consumption() + dof_handler.memory_consumption();

  double local_memory = memory_mf + memory_vectors + memory_grid;
  double global_memory = Utilities::MPI::sum(local_memory, MPI_COMM_WORLD);

  return global_memory / 1024.0 / 1024.0; // MB
}
