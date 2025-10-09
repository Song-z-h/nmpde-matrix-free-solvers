#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "Poisson3D_parallel.hpp"

// Main function.
int main(int argc, char *argv[])
{
  // This object calls MPI_Init when it is constructed, and MPI_Finalize when it
  // is destroyed. It also initializes several other libraries bundled with
  // dealii (e.g. p4est, PETSc, ...).
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const unsigned int degree = 1;
  const unsigned int mpi_rank =
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  std::vector<std::string> mesh_file_names = {
      "../mesh/mesh-square-5.msh",
      "../mesh/mesh-square-10.msh",
      "../mesh/mesh-square-20.msh",
      "../mesh/mesh-square-40.msh"};
  std::vector<int> mesh_Ns = {5, 10, 20, 40};

  ConvergenceTable table;

  std::ofstream convergence_file("convergence.csv");
  convergence_file << "h,eL2,eH1" << std::endl;

  for (unsigned int i = 0; i < mesh_Ns.size(); i++)
  {
    Poisson3DParallel problem(mesh_file_names[i], degree);
    problem.setup();
    problem.assemble();
    problem.solve();
    problem.output();

    const double h = 1.0 / (mesh_Ns[i]);
    const double error_L2 = problem.compute_error(VectorTools::L2_norm);
    const double error_H1 = problem.compute_error(VectorTools::H1_norm);

    table.add_value("h", h);
    table.add_value("L2", error_L2);
    table.add_value("H1", error_H1);

    convergence_file << h << "," << error_L2 << "," << error_H1 << std::endl;
  }
  if (mpi_rank == 0)
  {
    table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
    table.set_scientific("L2", true);
    table.set_scientific("H1", true);

    table.write_text(std::cout);
  }
  return 0;
}