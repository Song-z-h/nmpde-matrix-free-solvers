#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>

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
  const unsigned int mpi_rank =
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  std::vector<std::string> mesh_file_names;
  ConditionalOStream pcout(std::cout, mpi_rank == 0);

  if (Poisson3DParallel::dim == 3)
  {
    mesh_file_names = {
        "../mesh/mesh-cube-5.msh",
        "../mesh/mesh-cube-10.msh",
        "../mesh/mesh-cube-20.msh",
        "../mesh/mesh-cube-40.msh"};
  }
  if (Poisson3DParallel::dim == 2)
  {
    mesh_file_names = {
        "../mesh/mesh-square-5.msh",
        "../mesh/mesh-square-10.msh",
        "../mesh/mesh-square-20.msh",
        "../mesh/mesh-square-40.msh"};
  }
  std::vector<int> mesh_Ns = {5, 10, 20, 40};

  // Initialize TimerOutput for summary output
  TimerOutput timer(MPI_COMM_WORLD, pcout, TimerOutput::summary, TimerOutput::wall_times);

  // Individual Timer objects for each section
  Timer setup_timer(MPI_COMM_WORLD);
  Timer assemble_timer(MPI_COMM_WORLD);
  Timer solve_timer(MPI_COMM_WORLD);
  Timer output_timer(MPI_COMM_WORLD);
  Timer error_timer(MPI_COMM_WORLD);

  ConvergenceTable table;

  std::ofstream convergence_file("convergence.csv");
  if (mpi_rank == 0)
  {
    convergence_file << "h,eL2,eH1,setup_time,assemble_time,solve_time,output_time,error_time" << std::endl;
  }

  for (unsigned int i = 0; i < mesh_Ns.size(); i++)
  {

    pcout << "Mesh size " << mesh_Ns[i] << std::endl;

    Poisson3DParallel problem(mesh_file_names[i]);
    double setup_time, assemble_time, solve_time, output_time, error_time;

    {
      TimerOutput::Scope t(timer, "Setup");
      setup_timer.start();
      problem.setup();
      setup_time = setup_timer.wall_time();
      setup_timer.stop();
    }
    {
      TimerOutput::Scope t(timer, "Assemble");
      assemble_timer.start();
      problem.assemble();
      assemble_time = assemble_timer.wall_time();
      assemble_timer.stop();
    }
    {
      TimerOutput::Scope t(timer, "Solve");
      solve_timer.start();
      problem.solve();
      solve_time = solve_timer.wall_time();
      solve_timer.stop();
    }
    {
      TimerOutput::Scope t(timer, "Output");
      output_timer.start();
      problem.output();
      output_time = output_timer.wall_time();
      output_timer.stop();
    }

    const double h = 1.0 / (mesh_Ns[i]);
    double error_L2, error_H1;
    {
      TimerOutput::Scope t(timer, "ErrorComputation");
      error_timer.start();
      error_L2 = problem.compute_error(VectorTools::L2_norm);
      error_H1 = problem.compute_error(VectorTools::H1_norm);
      error_time = error_timer.wall_time();
      error_timer.stop();
    }
    table.add_value("h", h);
    table.add_value("L2", error_L2);
    table.add_value("H1", error_H1);

    if (mpi_rank == 0)
    {
      convergence_file << h << "," << error_L2 << "," << error_H1 << ","
                       << setup_time << "," << assemble_time << "," << solve_time << ","
                       << output_time << "," << error_time << std::endl;
    }
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