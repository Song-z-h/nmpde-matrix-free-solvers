#include "Heat.hpp"
#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <iostream>
#include <vector>
// Main function.
int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
  ConvergenceTable table;

  const std::vector<double> deltat_vals = {0.1,
                                           0.05,
                                           0.025,
                                           0.0125};

  // const std::string  mesh_file_name = "../mesh/mesh-cube-20.msh";
  const unsigned int N = 500;
  const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  ConditionalOStream pcout(std::cout, mpi_rank == 0);

  const double T = 1.0;
  const double theta = 1.0;

    auto timer = std::make_shared<TimerOutput>(
      MPI_COMM_WORLD,
      pcout,
      TimerOutput::summary,
      TimerOutput::wall_times);

  Timer setup_timer(MPI_COMM_WORLD);
  Timer solve_timer(MPI_COMM_WORLD);
  Timer error_timer(MPI_COMM_WORLD);

  std::ofstream convergence_file;

  if (mpi_rank == 0)
  {
    convergence_file.open("convergence_time_dependent.csv");
    // Similar style as Poisson, but with deltat instead of h
    convergence_file
    << "deltat,eL2,eH1,setup_time,"
    << "total_linear_solve_time,avg_time_per_step,"
    << "avg_gmres_iters_per_step,dofs_per_second,"
    << "error_time,memory_MB"
    << std::endl;

  }

  for (unsigned int i = 0; i < deltat_vals.size(); ++i)
  {
    Heat problem(N, "", T, deltat_vals[i], theta);
    problem.set_timer(timer); // share the same TimerOutput

    pcout << "===============================================" << std::endl;
    pcout << "Running with Δt = " << deltat_vals[i] << std::endl;
    // using > 100 N, the spatial error is small, you will see that the temperal error dominates
    // using < 10, spatial error dominates, you will see that the error convergence is hardly satisfactory

    double setup_time = 0.0;
    double error_time = 0.0;

    // --- Setup ---
    {
      TimerOutput::Scope t(*timer, "Setup");
      setup_timer.restart();
      problem.setup();
      setup_timer.stop();
      setup_time = setup_timer.wall_time();
    }

    const double ndofs = problem.get_number_of_dofs();

    // --- Memory after setup ---
    double precise_memory_mb = problem.get_memory_consumption();
    Utilities::System::MemoryStats stats;
    Utilities::System::get_memory_stats(stats);
    pcout << "  > Precise Memory (MF + Vecs): " << precise_memory_mb << " MB" << std::endl;
    pcout << "  > System Peak RSS: " << stats.VmHWM / 1024.0 << " MB" << std::endl;

    // --- Time stepping (Heat::solve contains the time loop) ---
    {
      //solve_timer.restart();
      problem.solve();
      //solve_timer.stop();
      //solve_time = solve_timer.wall_time();
    }

     // --- High-level performance metrics (SC17-style) ---
    const unsigned int n_steps = problem.get_n_time_steps();
    const auto total_lin_time  = problem.get_total_linear_solve_time(); // [s]
    const auto total_iters     = problem.get_total_gmres_iterations();

    const double avg_time_per_step  = (n_steps > 0 ? total_lin_time / n_steps : 0.0);
    const double avg_iters_per_step = (n_steps > 0 ? static_cast<double>(total_iters) / n_steps : 0.0);
    const double avg_time_per_iter  = (total_iters > 0 ? total_lin_time / static_cast<double>(total_iters) : 0.0);

    // DoFs per second (global)
    const double dofs_per_second = (total_lin_time > 0.0
                                      ? (ndofs * static_cast<double>(n_steps)) / total_lin_time
                                      : 0.0);

    pcout << "  --- Performance summary for Δt = " << deltat_vals[i] << " ---\n"
          << "    n_time_steps        = " << n_steps << "\n"
          << "    total DoFs          = " << ndofs << "\n"
          << "    total linear time   = " << total_lin_time << " s\n"
          << "    avg time per step   = " << avg_time_per_step << " s\n"
          << "    total GMRES iters   = " << total_iters << "\n"
          << "    avg iters per step  = " << avg_iters_per_step << "\n"
          << "    avg time per iter   = " << avg_time_per_iter << " s\n"
          << "    DoFs per second     = " << dofs_per_second << " DoFs/s\n";

    // --- Error at final time T ---
    double error_L2 = 0.0, error_H1 = 0.0;
    {
      TimerOutput::Scope t(*timer, "ErrorComputation");
      error_timer.restart();
      error_L2 = problem.compute_error(VectorTools::L2_norm);
      error_H1 = problem.compute_error(VectorTools::H1_norm);
      error_timer.stop();
      error_time = error_timer.wall_time();
    }

    table.add_value("deltat", deltat_vals[i]);
    table.add_value("L2", error_L2);
    table.add_value("H1", error_H1);
    table.add_value("Memory", precise_memory_mb);
    if (mpi_rank == 0)
    {
       convergence_file << deltat_vals[i] << ","
                       << error_L2 << ","
                       << error_H1 << ","
                       << setup_time << ","
                       << total_lin_time << ","
                       << avg_time_per_step << ","
                       << avg_iters_per_step << ","
                       << dofs_per_second << ","
                       << error_time << ","
                       << precise_memory_mb
                       << std::endl;
    }
  }

  // Only for Exercise 1:
  // Evaluate slopes in log base 2:
  if (mpi_rank == 0)
  {
    convergence_file.close();

    table.evaluate_all_convergence_rates(
        ConvergenceTable::reduction_rate_log2);

    table.set_scientific("L2", true);
    table.set_scientific("H1", true);

    std::cout << "===============================================" << std::endl;
    table.write_text(std::cout);
  }

  // Print TimerOutput summary at the end
  //timer.print_summary();

  return 0;
}