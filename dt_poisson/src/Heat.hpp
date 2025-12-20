#ifndef HEAT_HPP
#define HEAT_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <fstream>
#include <iostream>

#include <memory>
#include <iomanip>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

using namespace dealii;

// Class representing the non-linear diffusion problem.
class Heat
{

public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

  // Function for the epsilon coefficient.
  class FunctionMu : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 1.0;
    }
  };
  // coeficient for reaction
  class FunctionK : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 1.0;
    }
  };
  class FunctionB : public Function<dim>
  {
  public:
    FunctionB(const double beta0_in = 0.0)
        : Function<dim>(), beta0(beta0_in)
    {
    }

    virtual double
    value(const Point<dim> &p,
          const unsigned int component = 0) const override
    {
      if (component == 0)
        return beta0 * (p[0] - 1.0); // b_x = beta0 (x-1)
      else
        return 0.0;
    }

    virtual void
    vector_value(const Point<dim> &p,
                 Vector<double> &values) const override
    {
      for (unsigned int d = 0; d < dim; ++d)
        values[d] = value(p, d);
    }

  private:
    double beta0;
  };

  // coefficient for dirichlet boundary values
  class FunctionBeta : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 1.0;
    }
  };

  class FunctionAlpha : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

  // Function for the forcing term: u_t - Δu + k u = f, with k = 1, μ = 1.
  class ForcingTerm : public Function<dim>
  {
  public:
    explicit ForcingTerm(const double beta0_in = 0.0)
        : Function<dim>(), beta0(beta0_in)
    {
    }

    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      const double t = this->get_time();
      const double G = std::sin(numbers::PI * t);
      const double G_t = numbers::PI * std::cos(numbers::PI * t);

      const double k = 1.0;
      const double beta = beta0;

      // dim == 3 in this project
      const double x = p[0];
      const double y = p[1];
      const double z = p[2];

      const double two_pi = 2.0 * numbers::PI;
      const double four_pi = 4.0 * numbers::PI;
      const double three_pi = 3.0 * numbers::PI;

      const double U =
          std::sin(two_pi * x) *
          std::sin(four_pi * y) *
          std::sin(three_pi * z);

      const double U_x =
          two_pi * std::cos(two_pi * x) *
          std::sin(four_pi * y) *
          std::sin(three_pi * z);

      const double lambda = 29.0 * numbers::PI * numbers::PI;

      // same formula as in the MF code
      return U * G_t + (lambda + k + beta) * U * G + beta * (x - 1.0) * U_x * G;
    }

  private:
    double beta0;
  };

  // Function for the initial condition.
  class FunctionU0 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

  // function for stabilization terms

  class FunctionTau : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      const int N = 10;
      const double h = 1.0 / N;
      const double theta = 1.0;
      const double k = 25.0;
      const double psi = 1.0 / tanh(theta) - 1.0 / theta;
      return h / (2.0 * k) * psi * (h * k) / 2.0;
    }
  };

  // Exact solution.
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution() = default;

    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      const double t = this->get_time();
      const double Gt = std::sin(M_PI * t);

      if (dim == 1)
      {
        const double x = p[0];
        const double U = std::sin(M_PI * x);
        return U * Gt;
      }
      else if (dim == 2)
      {
        const double x = p[0];
        const double y = p[1];
        const double U =
            std::sin(2.0 * M_PI * x) *
            std::sin(4.0 * M_PI * y);
        return U * Gt;
      }
      else if (dim == 3)
      {
        const double x = p[0];
        const double y = p[1];
        const double z = p[2];
        const double U =
            std::sin(2.0 * M_PI * x) *
            std::sin(4.0 * M_PI * y) *
            std::sin(3.0 * M_PI * z);
        return U * Gt;
      }

      // Fallback (should not happen for dim=1,2,3)
      return 0.0;
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> result;
      const double t = this->get_time();
      const double Gt = std::sin(M_PI * t);

      if (dim == 1)
      {
        const double x = p[0];
        result[0] = M_PI * std::cos(M_PI * x) * Gt;
      }
      else if (dim == 2)
      {
        const double x = p[0];
        const double y = p[1];

        result[0] =
            2.0 * M_PI * std::cos(2.0 * M_PI * x) *
            std::sin(4.0 * M_PI * y) * Gt;

        result[1] =
            4.0 * M_PI * std::sin(2.0 * M_PI * x) *
            std::cos(4.0 * M_PI * y) * Gt;
      }
      else if (dim == 3)
      {
        const double x = p[0];
        const double y = p[1];
        const double z = p[2];

        result[0] =
            2.0 * M_PI * std::cos(2.0 * M_PI * x) *
            std::sin(4.0 * M_PI * y) *
            std::sin(3.0 * M_PI * z) * Gt;

        result[1] =
            4.0 * M_PI * std::sin(2.0 * M_PI * x) *
            std::cos(4.0 * M_PI * y) *
            std::sin(3.0 * M_PI * z) * Gt;

        result[2] =
            3.0 * M_PI * std::sin(2.0 * M_PI * x) *
            std::sin(4.0 * M_PI * y) *
            std::cos(3.0 * M_PI * z) * Gt;
      }

      return result;
    }
  };

  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  Heat(const int N_,
       const std::string &mesh_file_name_,
       const unsigned int &r_,
       const double &T_,
       const double &deltat_,
       const double &theta_,
       const double beta0_)
      : b(beta0_), forcing_term(beta0_), N(N_), mesh_file_name(mesh_file_name_), r(r_),
        T(T_), deltat(deltat_), theta(theta_), time(0.0),
        mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
        mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
        pcout(std::cout, mpi_rank == 0), mesh(MPI_COMM_WORLD),
        beta0(beta0_)
  {
  }

  // Initialization.
  void
  setup();

  // Solve the problem.
  void
  solve();

  double
  compute_error(const VectorTools::NormType &norm_type);

  // Optional shared TimerOutput (non-owning)
  void set_timer(const std::shared_ptr<TimerOutput> &timer_)
  {
    timer = timer_;
  }

  // High-level performance stats (for convergence driver)
  unsigned int get_n_time_steps() const { return n_time_steps; }
  unsigned long long get_total_gmres_iterations() const
  {
    return total_gmres_iterations;
  }
  double get_total_linear_solve_time() const
  {
    return total_linear_solve_time;
  }

  // Number of degrees of freedom
  unsigned int get_number_of_dofs() const { return dof_handler.n_dofs(); }

  // Memory usage in MB
  double get_memory_consumption() const;

protected:
  // Assemble the mass and stiffness matrices.
  void
  assemble_matrices();

  // Assemble the right-hand side of the problem.
  void
  assemble_rhs(const double &time);

  // Solve the problem for one time step.
  void
  solve_time_step();

  // Output.
  void
  output(const unsigned int &time_step) const;

  // Problem definition. ///////////////////////////////////////////////////////

  // mu coefficient.
  FunctionMu mu;
  FunctionB b;
  // k coefficient
  FunctionK k;

  // exact solution
  ExactSolution exact_solution;

  // coef for dirichlet
  FunctionBeta beta;
  FunctionAlpha alpha;

  // stabilization term
  FunctionTau tau;

  // Forcing term.
  ForcingTerm forcing_term;

  // Initial condition.
  FunctionU0 u_0;

  // Discretization. ///////////////////////////////////////////////////////////

  // number of mesh points
  const unsigned int N;

  // Mesh file name.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Final time.
  const double T;

  // Time step.
  const double deltat;

  // Theta parameter of the theta method.
  const double theta;

  // current time
  double time;

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // quadrature boundary
  std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // Mass matrix M / deltat.
  TrilinosWrappers::SparseMatrix mass_matrix;

  // Stiffness matrix A.
  TrilinosWrappers::SparseMatrix stiffness_matrix;

  // Matrix on the left-hand side (M / deltat + theta A).
  TrilinosWrappers::SparseMatrix lhs_matrix;

  // Matrix on the right-hand side (M / deltat - (1 - theta) A).
  TrilinosWrappers::SparseMatrix rhs_matrix;

  // Right-hand side vector in the linear system.
  TrilinosWrappers::MPI::Vector system_rhs;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;

  // Boundary values map (added for fix).
  std::map<types::global_dof_index, double> boundary_values;

  AffineConstraints<double> constraints;

  std::shared_ptr<TimerOutput> timer;

  // advection strength used by b and f
  double beta0;

  // high-level performance counters
  unsigned int n_time_steps = 0;
  unsigned long long total_gmres_iterations = 0;
  double total_linear_solve_time = 0.0; // [s]
};

#endif
