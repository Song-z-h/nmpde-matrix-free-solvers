  #include "Heat.hpp"
  #include <deal.II/base/convergence_table.h>

  // Main function.
  int
  main(int argc, char *argv[])
  {
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
    ConvergenceTable table;

    const std::vector<double>      deltat_vals = {0.1,
                                            0.05,
                                            0.025,
                                            0.0125};


    //const std::string  mesh_file_name = "../mesh/mesh-cube-20.msh";
    const unsigned int degree         = 2;
    const unsigned int N = 300;

    const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);


    const double T      = 1.0;
    const double theta  = 0.0;

    std::ofstream convergence_file("convergence.csv");
    convergence_file << "deltat,eL2,eH1" << std::endl;

    for (unsigned int i = 0; i < deltat_vals.size(); ++i)
      {
        Heat problem(N, "", degree, T, deltat_vals[i], theta);
        //using > 100 N, the spatial error is small, you will see that the temperal error dominates
        //using < 10, spatial error dominates, you will see that the error convergence is hardly satisfactory

        problem.setup();
        problem.solve();

        // Only for Exercise 1:
        const double error_L2 = problem.compute_error(VectorTools::L2_norm);
        const double error_H1 = problem.compute_error(VectorTools::H1_norm);

        table.add_value("deltat", deltat_vals[i]);
        table.add_value("L2", error_L2);
        table.add_value("H1", error_H1);
        convergence_file << deltat_vals[i] << "," << error_L2 << "," << error_H1
                        << std::endl;
      }

    // Only for Exercise 1:
    // Evaluate slopes in log base 2:
    if(mpi_rank == 0){
      table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
      table.set_scientific("L2", true);
      table.set_scientific("H1", true);
      table.write_text(std::cout);
    }
    return 0;
  }