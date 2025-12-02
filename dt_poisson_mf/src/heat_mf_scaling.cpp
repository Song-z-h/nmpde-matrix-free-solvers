// heat_mf_scaling.cpp
#include "Heat.hpp"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <cstdlib>

using namespace dealii;

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  const unsigned int mpi_rank =
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const unsigned int mpi_size =
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  ConditionalOStream pcout(std::cout, mpi_rank == 0);

  // -----------------------------
  // Simple command line parsing:
  //   argv[1] = N (subdivisions)
  //   argv[2] = dt
  //   argv[3] = T
  //   argv[4] = theta
  // If not given, use defaults.
  // -----------------------------
  unsigned int N     = 64;
  double       dt    = 1e-2;
  double       T     = 1.0;
  double       theta = 1.0;

  if (argc > 1)
    N = static_cast<unsigned int>(std::atoi(argv[1]));
  if (argc > 2)
    dt = std::atof(argv[2]);
  if (argc > 3)
    T = std::atof(argv[3]);
  if (argc > 4)
    theta = std::atof(argv[4]);

  using PreType = Heat::PreconditionerType;

  auto timer = std::make_shared<TimerOutput>(
    MPI_COMM_WORLD,
    pcout,
    TimerOutput::summary,
    TimerOutput::wall_times);

  // -----------------------------
  // Problem setup
  // -----------------------------
  Heat problem(N, "", T, dt, theta, PreType::Jacobi);
  problem.set_timer(timer);

  Timer setup_timer(MPI_COMM_WORLD);
  setup_timer.restart();
  {
    TimerOutput::Scope t(*timer, "Setup");
    problem.setup();
  }
  setup_timer.stop();
  const double setup_time = setup_timer.wall_time();

  const double ndofs      = problem.get_number_of_dofs();
  const double mem_MB     = problem.get_memory_consumption();

  // -----------------------------
  // Time stepping + linear solves
  // -----------------------------
  Timer total_timer(MPI_COMM_WORLD);
  total_timer.restart();
  problem.solve();
  total_timer.stop();
  const double total_wall_time = total_timer.wall_time();

  const unsigned int       n_steps        = problem.get_n_time_steps();
  const double             total_lin_time = problem.get_total_linear_solve_time();
  const unsigned long long total_iters    = problem.get_total_gmres_iterations();

  const double dofs_per_second =
    (total_lin_time > 0.0 ? (ndofs * static_cast<double>(n_steps)) /
                             total_lin_time
                          : 0.0);

  if (mpi_rank == 0)
  {
    // Print CSV header (you can delete this line later if you don’t want it)
    /*std::cout
      << "backend,mpi_size,N,ndofs,T,dt,theta,"
      << "setup_time,total_wall_time,total_linear_solve_time,"
      << "n_time_steps,total_gmres_iters,dofs_per_second,memory_MB\n";*/

    std::cout << "mf,"
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
              << mem_MB
              << std::endl;
  }

  return 0;
}
