#include "Heat.hpp"

#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <iostream>
#include <memory>

using namespace dealii;

int main(int argc, char *argv[])
{
  // Initialize MPI
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const unsigned int mpi_size = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  ConditionalOStream pcout(std::cout, mpi_rank == 0);

  // ---------------------------------------------------------------------------
  // Parse command line: N dt T theta
  // ---------------------------------------------------------------------------
  if (argc != 5)
  {
    if (mpi_rank == 0)
      std::cerr << "Usage: " << argv[0] << " N dt T theta\n";
    return 1;
  }

  const unsigned int N     = static_cast<unsigned int>(std::stoul(argv[1]));
  const double       dt    = std::stod(argv[2]);
  const double       T     = std::stod(argv[3]);
  const double       theta = std::stod(argv[4]);

  const unsigned int degree = 3; // same polynomial degree as before

  // ---------------------------------------------------------------------------
  // Shared TimerOutput (like in the MF code)
  // ---------------------------------------------------------------------------
  auto timer = std::make_shared<TimerOutput>(
      MPI_COMM_WORLD,
      pcout,
      TimerOutput::summary,
      TimerOutput::wall_times);

  // Measure total wall time
  const double t_wall_start = MPI_Wtime();

  // ---------------------------------------------------------------------------
  // Setup phase
  // ---------------------------------------------------------------------------
  Timer setup_timer(MPI_COMM_WORLD);
  setup_timer.restart();

  Heat problem(N, "", degree, T, dt, theta, 1.0);
  problem.set_timer(timer);
  problem.setup();

  setup_timer.stop();
  const double setup_time = setup_timer.wall_time();

  const double ndofs            = problem.get_number_of_dofs();
  const double precise_memory_mb = Utilities::MPI::sum(problem.get_process_rss_MB(), MPI_COMM_WORLD);


  pcout << "  > Precise Memory (Sparse + Vecs): "
        << precise_memory_mb << " MB\n";

  // ---------------------------------------------------------------------------
  // Time stepping
  // ---------------------------------------------------------------------------
  problem.solve();

  const double total_wall_time = MPI_Wtime() - t_wall_start;

  // ---------------------------------------------------------------------------
  // Performance metrics
  // ---------------------------------------------------------------------------
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

  pcout << "  --- Performance summary (Sparse, Jacobi) ---\n"
        << "    mpi_size           = " << mpi_size << "\n"
        << "    N                  = " << N << "\n"
        << "    T                  = " << T << "\n"
        << "    dt                 = " << dt << "\n"
        << "    theta              = " << theta << "\n"
        << "    n_time_steps       = " << n_steps << "\n"
        << "    total DoFs         = " << ndofs << "\n"
        << "    total wall time    = " << total_wall_time << " s\n"
        << "    setup time         = " << setup_time << " s\n"
        << "    total linear time  = " << total_lin_time << " s\n"
        << "    avg time per step  = " << avg_time_per_step << " s\n"
        << "    total GMRES iters  = " << total_iters << "\n"
        << "    avg iters / step   = " << avg_iters_per_step << "\n"
        << "    avg time / iter    = " << avg_time_per_iter << " s\n"
        << "    DoFs per second    = " << dofs_per_second << " DoFs/s\n"
        << "    memory             = " << precise_memory_mb << " MB\n";

  // ---------------------------------------------------------------------------
  // CSV line for post-processing (same style as MF scaling)
  // ---------------------------------------------------------------------------
  if (mpi_rank == 0)
  {
    // Header (you can comment this out if you call the binary many times
    // and concatenate the outputs)
    std::cout << "backend,mpi_size,N,ndofs,T,dt,theta,"
              << "setup_time,total_wall_time,total_linear_solve_time,"
              << "n_time_steps,total_gmres_iters,dofs_per_second,memory_MB\n";

    std::cout << "sparse,"
              << mpi_size << ","
              << N << ","
              << ndofs << ","
              << T << ","
              << dt << ","
              << theta << ","
              << setup_time << ","
              << total_wall_time << ","
              << total_lin_time << ","
              << n_steps << ","
              << total_iters << ","
              << dofs_per_second << ","
              << precise_memory_mb << "\n";
  }

  return 0;
}
