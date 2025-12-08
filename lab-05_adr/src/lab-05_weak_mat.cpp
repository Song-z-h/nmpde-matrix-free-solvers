#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib> // std::atoi
#include <cmath>   // std::cbrt, std::round

#include "Poisson3D_parallel.hpp"

using namespace dealii;

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const unsigned int nprocs   = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  ConditionalOStream pcout(std::cout, mpi_rank == 0);

  // ----------------------------
  // Weak-scaling parameters
  // ----------------------------
  // N0 = base subdivisions per axis for 1 process
  int N0 = 40;
  if (argc > 1)
    N0 = std::atoi(argv[1]);

  const unsigned int degree = 3; // must match MF fe_degree for fair comparison

  std::string csv_name = "weak_scaling_mat.csv";
  if (argc > 2 && mpi_rank == 0)
    csv_name = argv[2];

  // Choose N(p) ~ N0 * p^(1/3) so that total cells ~ constant per rank
  const double scale = std::cbrt(static_cast<double>(nprocs));
  int N = static_cast<int>(std::round(N0 * scale));
  if (N < 2)
    N = 2;

  pcout << "Weak scaling run (matrix-based):" << std::endl;
  pcout << "  N0     = " << N0 << std::endl;
  pcout << "  nprocs = " << nprocs << std::endl;
  pcout << "  scale  = " << scale << std::endl;
  pcout << "  N      = " << N << "  (target cells per rank ~ constant)" << std::endl;

  std::vector<int> mesh_Ns = {N};

  TimerOutput timer(MPI_COMM_WORLD,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times);

  Timer setup_timer(MPI_COMM_WORLD);
  Timer assemble_timer(MPI_COMM_WORLD);
  Timer solve_timer(MPI_COMM_WORLD);
  Timer output_timer(MPI_COMM_WORLD);
  Timer error_timer(MPI_COMM_WORLD);

  ConvergenceTable table;

  std::ofstream convergence_file;
  if (mpi_rank == 0)
  {
    convergence_file.open(csv_name);
    convergence_file
      << "nprocs,N0,N,h,ndofs,eL2,eH1,"
      << "setup_time,assemble_time,solve_time,output_time,error_time,total_time,"
      << "memory_MB,memory_MB_per_dof,"
      << "cg_iters,avg_time_per_iter,"
      << "dofs_per_second,million_dofs_per_second,dofs_per_second_per_core"
      << std::endl;
  }

  for (unsigned int i = 0; i < mesh_Ns.size(); ++i)
  {
    const int N_current = mesh_Ns[i];
    pcout << "-------------------------------------------------" << std::endl;
    pcout << "Mesh subdivisions N = " << N_current << std::endl;

    Poisson3DParallelMf problem(N_current, degree);

    double setup_time    = 0.0;
    double assemble_time = 0.0;
    double solve_time    = 0.0;
    double output_time   = 0.0;
    double error_time    = 0.0;

    // Setup
    {
      TimerOutput::Scope t(timer, "Setup");
      setup_timer.start();
      problem.setup();
      setup_time = setup_timer.wall_time();
      setup_timer.stop();
    }

    // Memory
    const double precise_memory_mb = problem.get_memory_consumption();
    Utilities::System::MemoryStats stats;
    Utilities::System::get_memory_stats(stats);

    pcout << "  > Precise Memory (mat + vecs): " << precise_memory_mb << " MB" << std::endl;
    pcout << "  > System Peak RSS:            " << stats.VmHWM / 1024.0 << " MB" << std::endl;

    // Assemble
    {
      TimerOutput::Scope t(timer, "Assemble");
      assemble_timer.start();
      problem.assemble();
      assemble_time = assemble_timer.wall_time();
      assemble_timer.stop();
    }

    // Solve
    {
      TimerOutput::Scope t(timer, "Solve");
      solve_timer.start();
      problem.solve();
      solve_time = solve_timer.wall_time();
      solve_timer.stop();
    }

    // Output
    {
      TimerOutput::Scope t(timer, "Output");
      output_timer.start();
      problem.output();
      output_time = output_timer.wall_time();
      output_timer.stop();
    }

    // Error
    const double h = 1.0 / static_cast<double>(N_current);
    double error_L2 = 0.0, error_H1 = 0.0;
    {
      TimerOutput::Scope t(timer, "ErrorComputation");
      error_timer.start();
      error_L2  = problem.compute_error(VectorTools::L2_norm);
      error_H1  = problem.compute_error(VectorTools::H1_norm);
      error_time = error_timer.wall_time();
      error_timer.stop();
    }

    // HPC metrics
    const double       ndofs    = problem.get_number_of_dofs();
    const unsigned int cg_iters = problem.get_last_cg_iterations();

    const double total_time =
      setup_time + assemble_time + solve_time + output_time + error_time;

    double dofs_per_second           = 0.0;
    double avg_time_per_iter         = 0.0;
    double million_dofs_per_second   = 0.0;
    double dofs_per_second_per_core  = 0.0;

    if (solve_time > 0.0 && cg_iters > 0)
    {
      dofs_per_second          = (ndofs * static_cast<double>(cg_iters)) / solve_time;
      avg_time_per_iter        = solve_time / static_cast<double>(cg_iters);
      million_dofs_per_second  = dofs_per_second / 1e6;
      dofs_per_second_per_core = dofs_per_second / static_cast<double>(nprocs);
    }

    const double memory_per_dof =
      (ndofs > 0.0) ? (precise_memory_mb / ndofs) : 0.0;

    table.add_value("h", h);
    table.add_value("ndofs", ndofs);
    table.add_value("L2", error_L2);
    table.add_value("H1", error_H1);
    table.add_value("Memory_MB", precise_memory_mb);
    table.add_value("Mem/DoF", memory_per_dof);
    table.add_value("CG_iters", cg_iters);
    table.add_value("DoFs/s[1e6]", million_dofs_per_second);

    if (mpi_rank == 0)
    {
      convergence_file << nprocs << ","
                       << N0 << ","
                       << N_current << ","
                       << h << ","
                       << ndofs << ","
                       << error_L2 << ","
                       << error_H1 << ","
                       << setup_time << ","
                       << assemble_time << ","
                       << solve_time << ","
                       << output_time << ","
                       << error_time << ","
                       << total_time << ","
                       << precise_memory_mb << ","
                       << memory_per_dof << ","
                       << cg_iters << ","
                       << avg_time_per_iter << ","
                       << dofs_per_second << ","
                       << million_dofs_per_second << ","
                       << dofs_per_second_per_core
                       << std::endl;
    }
  }

  if (mpi_rank == 0)
  {
    convergence_file.close();
    table.set_scientific("L2", true);
    table.set_scientific("H1", true);
    table.write_text(std::cout);
  }

  return 0;
}
