#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <deal.II/base/utilities.h>
#include "Poisson3D_parallel.hpp"

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  ConditionalOStream pcout(std::cout, mpi_rank == 0);

  // Define file names and Ns
  const unsigned int degree = 3;
  std::vector<int> mesh_Ns = {5, 10, 20, 40};

  ConvergenceTable table;
  TimerOutput timer(MPI_COMM_WORLD, pcout, TimerOutput::summary, TimerOutput::wall_times);

  // Timers
  Timer setup_timer(MPI_COMM_WORLD);
  Timer assemble_timer(MPI_COMM_WORLD);
  Timer solve_timer(MPI_COMM_WORLD);
  Timer output_timer(MPI_COMM_WORLD);
  Timer error_timer(MPI_COMM_WORLD);

  // Prepare CSV output file
  std::ofstream convergence_file;
  if (mpi_rank == 0)
  {
    convergence_file.open("convergence.csv");
    // <--- NEW: Added "memory_MB" to the header
convergence_file
  << "h,ndofs,eL2,eH1,"
  << "setup_time,assemble_time,solve_time,output_time,error_time,total_time,"
  << "memory_MB,memory_MB_per_dof,"
  << "cg_iters,avg_time_per_iter,"
  << "dofs_per_second,million_dofs_per_second,dofs_per_second_per_core"
  << std::endl;
  }

  for (unsigned int i = 0; i < mesh_Ns.size(); i++)
  {
    pcout << "Mesh refinement " << mesh_Ns[i] << std::endl;

    // 1. Create Problem
    Poisson3DParallelMf problem(mesh_Ns[i], degree, 4.0);

    double setup_time, assemble_time, solve_time, output_time, error_time;

    // 2. Setup (Allocates Matrix/Vectors)
    {
      TimerOutput::Scope t(timer, "Setup");
      setup_timer.start();
      problem.setup();
      setup_time = setup_timer.wall_time();
      setup_timer.stop();
    }

    // ============================================================
    // 3. MEASURE MEMORY (NEW SECTION)
    // ============================================================
    // Note: You must have implemented get_memory_consumption() in Poisson3D_parallel.cpp
    double precise_memory_mb = problem.get_memory_consumption(); 

    // Optional: Get System RSS for printout only
    Utilities::System::MemoryStats stats;
    Utilities::System::get_memory_stats(stats);
    
    pcout << "  > Precise Memory: " << precise_memory_mb << " MB" << std::endl;
    // ============================================================

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

// --- NEW: HPC metrics ---
const double       ndofs    = problem.get_number_of_dofs();
const unsigned int cg_iters = problem.get_last_cg_iterations();

// total wall time (just from the timers in main)
const double total_time =
  setup_time + assemble_time + solve_time + output_time + error_time;

double dofs_per_second        = 0.0;
double avg_time_per_iter      = 0.0;
double million_dofs_per_second = 0.0;
double dofs_per_second_per_core = 0.0;

if (solve_time > 0.0 && cg_iters > 0)
{
  // total work ~ ndofs * iterations (for CG with matrix-based operator)
  dofs_per_second = (ndofs * static_cast<double>(cg_iters)) / solve_time;
  avg_time_per_iter = solve_time / static_cast<double>(cg_iters);
  million_dofs_per_second = dofs_per_second / 1e6;
  dofs_per_second_per_core =
    dofs_per_second / Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
}

// memory per DoF (MB/DoF)
const double memory_per_dof = (ndofs > 0.0) ? (precise_memory_mb / ndofs) : 0.0;

// Add to console ConvergenceTable (you can adjust which fields you want here)
table.add_value("h", h);
table.add_value("ndofs", ndofs);
table.add_value("L2", error_L2);
table.add_value("H1", error_H1);
table.add_value("Memory_MB", precise_memory_mb);
table.add_value("Mem/DoF", memory_per_dof);
table.add_value("CG_iters", cg_iters);
table.add_value("DoFs/s[1e6]", million_dofs_per_second);

// --- CSV output ---
if (mpi_rank == 0)
{
  convergence_file << h << ","
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
    convergence_file.close(); // Close file

    // Print pretty table to console
    table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
    table.set_scientific("L2", true);
    table.set_scientific("H1", true);
    table.write_text(std::cout);
  }

  return 0;
}