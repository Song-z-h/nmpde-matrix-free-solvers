#include "Poisson3D_parallel.hpp"

void Poisson3DParallel::setup()
{
  pcout << "===============================================" << std::endl;

  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    // First we read the mesh from file into a serial (i.e. not parallel)
    // triangulation.
    Triangulation<dim> mesh_serial;

    {
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(mesh_serial);

      std::ifstream grid_in_file(mesh_file_name);
      grid_in.read_msh(grid_in_file);
    }

    // Then, we copy the triangulation into the parallel one.
    {
      GridTools::partition_triangulation(mpi_size, mesh_serial);
      const auto construction_data = TriangulationDescription::Utilities::
          create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
      mesh.create_triangulation(construction_data);
    }

    // Notice that we write here the number of *global* active cells (across all
    // processes).
    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space. This is the same as in serial codes.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    // We retrieve the set of locally owned DoFs, which will be useful when
    // initializing linear algebra classes.
    locally_owned_dofs = dof_handler.locally_owned_dofs();

    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    // Build constraints here
    pcout << "Building the constraints" << std::endl;
    constraints.clear();
    //constraints.reinit(locally_owned_dofs);
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

    // Finally, we initialize the right-hand side and solution vectors.
    // pcout << "  Initializing the system right-hand side" << std::endl;
    // system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    // pcout << "  Initializing the solution vector" << std::endl;
    // solution.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  }

  // matrix-free settings
  {

    additional_data.tasks_parallel_scheme =
        MatrixFree<dim, double>::AdditionalData::none; // single-threaded, keep MPI
    additional_data.mapping_update_flags =
        (update_gradients | update_values | update_JxW_values | update_quadrature_points);

    MappingFE<dim> mapping_local(*fe);
    mf_storage = std::make_shared<MatrixFree<dim, Number>>();
    mf_storage->reinit(mapping_local, dof_handler, constraints,
                       QGaussSimplex<dim>(fe->degree + 1),
                       additional_data);

    // Initialize vectors using MatrixFree's partitioner for compatibility
    pcout << "  Initializing the system right-hand side" << std::endl;
    mf_storage->initialize_dof_vector(system_rhs);
    pcout << "  Initializing the solution vector" << std::endl;
    mf_storage->initialize_dof_vector(solution);

    // Initialize our matrix-free operator
    mf_operator.initialize(mf_storage);
    // mf_operator.set_diffusion(diffusion_coefficient);
    // mf_operator.set_reaction(reaction_coefficient);
    mf_operator.evaluate_coefficient(diffusion_coefficient, reaction_coefficient);
    //mf_operator.set_constraints(constraints);


    pcout << "  Initializing diagonal preconditioner" << std::endl;
    preconditioner.initialize(*mf_storage, mf_operator);
    pcout << "Setup completed" << std::endl;
  }
}

void Poisson3DParallel::assemble()
{
  pcout << "===============================================" << std::endl;

  pcout << "Assembling RHS (kept FEValues style for simplicity)" << std::endl;
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  MappingFE<dim> mapping_assemble(*fe);
  FEValues<dim> fe_values(mapping_assemble, *fe,
                          *quadrature,
                          update_values |
                              update_quadrature_points | update_JxW_values);

  // FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  // system_matrix = 0.0;
  system_rhs = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    // If current cell is not owned locally, we skip it.
    if (!cell->is_locally_owned())
      continue;

    // On all other cells (which are owned by current process), we perform the
    // assembly as usual.

    fe_values.reinit(cell);

    // cell_matrix = 0.0;
    cell_rhs = 0.0;

    for (unsigned int q = 0; q < n_q; ++q)
    {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {

        cell_rhs(i) += forcing_term.value(fe_values.quadrature_point(q)) *
                       fe_values.shape_value(i, q) * fe_values.JxW(q);
      }
    }

    cell->get_dof_indices(dof_indices);
    // system_matrix.add(dof_indices, cell_matrix);
    constraints.distribute_local_to_global(cell_rhs, dof_indices, system_rhs);
  }

  // Each process might have written to some rows it does not own (for instance,
  // if it owns elements that are adjacent to elements owned by some other
  // process). Therefore, at the end of the assembly, processes need to exchange
  // information: the compress method allows to do this.
  // system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}

void Poisson3DParallel::solve()
{
  pcout << "===============================================" << std::endl;

  pcout << "Solving with matrix-free operator (CG precond)" << std::endl;
  constraints.set_zero(system_rhs);
  SolverControl solver_control(10000, 1e-8 * system_rhs.l2_norm());
  SolverCG<VectorType> solver(solver_control);

  // PreconditionIdentity preconditioner;

  // initial guess zero
  solution = 0;

  solver.solve(mf_operator, solution, system_rhs, preconditioner);
  constraints.distribute(solution);

  pcout << "  CG iterations: " << solver_control.last_step() << "cg residuals: " << solver_control.last_value() << std::endl;
}

void Poisson3DParallel::output() const
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
  solution_ghost.update_ghost_values();  // FIXED: Update ghosts
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

  const std::filesystem::path mesh_path(mesh_file_name);
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
Poisson3DParallel::compute_error(const VectorTools::NormType &norm_type) const
{
  FE_SimplexP<dim> fe_linear(1);
  MappingFE mapping(fe_linear);
  // The error is an integral, and we approximate that integral using a
  // quadrature formula. To make sure we are accurate enough, we use a
  // quadrature formula with one node more than what we used in assembly.
  const QGaussSimplex<dim> quadrature_error(r + 2);

  VectorType ghosted_solution(locally_owned_dofs,
                              locally_relevant_dofs,
                              MPI_COMM_WORLD);
  ghosted_solution = solution;
  ghosted_solution.update_ghost_values();  // FIXED: Update ghosts
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