#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <deal.II/base/utilities.h> // Needed for System Memory stats

#include "Poisson3D_parallel_mf.hpp"

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  ConditionalOStream pcout(std::cout, mpi_rank == 0);
  
  std::vector<int> mesh_Ns = {5, 10, 20, 40};

  TimerOutput timer(MPI_COMM_WORLD, pcout, TimerOutput::summary, TimerOutput::wall_times);

  Timer setup_timer(MPI_COMM_WORLD);
  Timer assemble_timer(MPI_COMM_WORLD);
  Timer solve_timer(MPI_COMM_WORLD);
  Timer output_timer(MPI_COMM_WORLD);
  Timer error_timer(MPI_COMM_WORLD);

  ConvergenceTable table;

  std::ofstream convergence_file;
  if (mpi_rank == 0)
  {
    convergence_file.open("convergence.csv");
    // <--- NEW: Added "memory_MB" to header
    convergence_file << "h,eL2,eH1,setup_time,assemble_time,solve_time,output_time,error_time,memory_MB" << std::endl;
  }

  for (unsigned int i = 0; i < mesh_Ns.size(); i++)
  {
    pcout << "Mesh size " << mesh_Ns[i] << std::endl;

    Poisson3DParallelMf problem(mesh_Ns[i]); 
    
    double setup_time, assemble_time, solve_time, output_time, error_time;

    {
      TimerOutput::Scope t(timer, "Setup");
      setup_timer.start();
      problem.setup();
      setup_time = setup_timer.wall_time();
      setup_timer.stop();
    }

    double precise_memory_mb = problem.get_memory_consumption();

    Utilities::System::MemoryStats stats;
    Utilities::System::get_memory_stats(stats);
    
    pcout << "  > Precise Memory (MF + Vecs): " << precise_memory_mb << " MB" << std::endl;
    pcout << "  > System Peak RSS: " << stats.VmHWM / 1024.0 << " MB" << std::endl;

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
    table.add_value("Memory", precise_memory_mb); 

    if (mpi_rank == 0)
    {
      // <--- NEW: Write memory to CSV
      convergence_file << h << "," 
                       << error_L2 << "," 
                       << error_H1 << ","
                       << setup_time << "," 
                       << assemble_time << "," 
                       << solve_time << ","
                       << output_time << "," 
                       << error_time << ","
                       << precise_memory_mb << std::endl; 
    }
  }
  
  if (mpi_rank == 0)
  {
    convergence_file.close();
    table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
    table.set_scientific("L2", true);
    table.set_scientific("H1", true);
    table.write_text(std::cout);
  }
  return 0;
}