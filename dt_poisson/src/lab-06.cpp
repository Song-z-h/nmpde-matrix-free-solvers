#include "Heat.hpp"

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <memory>

using namespace dealii;

struct RunConfig
{
  std::string label;    // for console output
  std::string filename; // CSV filename
};

void run_time_convergence(const RunConfig &config,
                          const std::vector<double> &deltat_vals,
                          const unsigned int N,
                          const unsigned int degree,
                          const double T,
                          const double theta,
                          const unsigned int mpi_rank,
                          ConditionalOStream &pcout,
                          const std::shared_ptr<TimerOutput> &timer)
{
  ConvergenceTable table;

  std::ofstream convergence_file;
  if (mpi_rank == 0)
  {
    convergence_file.open(config.filename);
    convergence_file
      << "deltat,eL2,eH1,setup_time,"
      << "total_linear_solve_time,avg_time_per_step,"
      << "avg_gmres_iters_per_step,dofs_per_second,"
      << "error_time,memory_MB"
      << std::endl;
  }

  Timer setup_timer(MPI_COMM_WORLD);
  Timer error_timer(MPI_COMM_WORLD);

  for (double deltat : deltat_vals)
  {
    Heat problem(N, "", degree, T, deltat, theta);
    problem.set_timer(timer);

    pcout << "===============================================" << std::endl;
    pcout << "Run = " << config.label << ", Δt = " << deltat << std::endl;

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
    pcout << "  > Precise Memory (matrices + vecs): "
          << precise_memory_mb << " MB" << std::endl;
    pcout << "  > System Peak RSS: "
          << stats.VmHWM / 1024.0 << " MB" << std::endl;

    // --- Time stepping ---
    problem.solve();

    // --- Performance metrics ---
    const unsigned int n_steps        = problem.get_n_time_steps();
    const double       total_lin_time = problem.get_total_linear_solve_time();
    const auto         total_iters    = problem.get_total_gmres_iterations();

    const double avg_time_per_step  =
      (n_steps > 0 ? total_lin_time / n_steps : 0.0);
    const double avg_iters_per_step =
      (n_steps > 0 ? static_cast<double>(total_iters) / n_steps : 0.0);
    const double avg_time_per_iter  =
      (total_iters > 0 ? total_lin_time / static_cast<double>(total_iters) : 0.0);
    const double dofs_per_second    =
      (total_lin_time > 0.0
       ? (ndofs * static_cast<double>(n_steps)) / total_lin_time
       : 0.0);

    pcout << "  --- Performance summary for Δt = " << deltat << " ---\n"
          << "    n_time_steps        = " << n_steps << "\n"
          << "    total DoFs          = " << ndofs << "\n"
          << "    total linear time   = " << total_lin_time << " s\n"
          << "    avg time per step   = " << avg_time_per_step << " s\n"
          << "    total GMRES iters   = " << total_iters << "\n"
          << "    avg iters per step  = " << avg_iters_per_step << "\n"
          << "    avg time per iter   = " << avg_time_per_iter << "\n"
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

    table.add_value("deltat", deltat);
    table.add_value("L2",     error_L2);
    table.add_value("H1",     error_H1);
    table.add_value("Memory", precise_memory_mb);

    if (mpi_rank == 0)
    {
      convergence_file << deltat << ","
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

  if (mpi_rank == 0)
  {
    convergence_file.close();
    table.evaluate_all_convergence_rates(
      ConvergenceTable::reduction_rate_log2);
    table.set_scientific("L2", true);
    table.set_scientific("H1", true);
    std::cout << "===============================================" << std::endl;
    std::cout << "Convergence table for " << config.label << ":\n";
    table.write_text(std::cout);
  }
}

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  ConditionalOStream pcout(std::cout, mpi_rank == 0);

  const unsigned int degree = 2;
  const unsigned int N      = 500;
  const double       T      = 1.0;
  const double       theta  = 1.0;

   const std::vector<double> deltat_vals = {0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125};

  auto timer = std::make_shared<TimerOutput>(
    MPI_COMM_WORLD,
    pcout,
    TimerOutput::summary,
    TimerOutput::wall_times);

  std::vector<RunConfig> runs = {
    {"Sparse_Jacobi", "convergence_sparse_jacobi.csv"}
  };

  for (const auto &run : runs)
    run_time_convergence(run,
                         deltat_vals,
                         N,
                         degree,
                         T,
                         theta,
                         mpi_rank,
                         pcout,
                         timer);

  return 0;
}
