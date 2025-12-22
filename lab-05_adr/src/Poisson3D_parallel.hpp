#ifndef POISSON_3D_HPP
#define POISSON_3D_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace dealii;

/**
 * Class managing the differential problem.
 */
class Poisson3DParallelMf
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

  // Diffusion coefficient.
  // In deal.ii, functions are implemented by deriving the dealii::Function
  // class, which provides an interface for the computation of function values
  // and their derivatives.
  class DiffusionCoefficient : public Function<dim>
  {
  public:
    // Constructor.
    DiffusionCoefficient()
    {
    }

    // Evaluation.
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 1.0;
    }
  };

  // Reaction coefficient.
  class ReactionCoefficient : public Function<dim>
  {
  public:
    // Constructor.
    ReactionCoefficient()
    {
    }

    // Evaluation.
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 1.0;
    }
  };

  // Advection coefficient: b = (beta0*(x-1), 0, 0)^T
  class AdvectionCoefficient : public Function<dim>
  {
  public:
    AdvectionCoefficient(const double beta0_in = 0.0)
        : Function<dim>(), beta0(beta0_in)
    {
    }

    virtual double
    value(const Point<dim> &p,
          const unsigned int component = 0) const override
    {
      if (component == 0)
        return beta0 * (p[0] - 1.0);
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

  // Forcing term consistent with: -div(mu∇u - b u) + k u = f
  class ForcingTerm : public Function<dim>
  {
  public:
    explicit ForcingTerm(const double beta0_in = 0.0)
        : Function<dim>(), beta0(beta0_in)
    {
    }

    double value(const Point<dim> &p,
                 const unsigned int = 0) const override
    {
      return value<double>(p);
    }

    template <typename number>
    number value(const Point<dim, number> &p,
                 const unsigned int = 0) const
    {
      // mu = 1, k = 1, b = (beta0*(x-1), 0, 0)
      const number x = p[0];

      if constexpr (dim == 2)
      {
        const number y = p[1];
        const number two_pi = number(2.0 * M_PI);
        const number four_pi = number(4.0 * M_PI);

        const number u =
            std::sin(two_pi * x) *
            std::sin(four_pi * y);

        const number ux =
            two_pi * std::cos(two_pi * x) *
            std::sin(four_pi * y);

        const number lambda = number(20.0 * M_PI * M_PI); // -Δu = λ u
        const number k = number(1.0);
        const number beta = number(beta0);

        // f = -Δu + div(b u) + k u
        //   = (λ + k + beta) u + beta (x-1) ux
        return (lambda + k + beta) * u + beta * (x - number(1.0)) * ux;
      }
      else if constexpr (dim == 3)
      {
        const number y = p[1];
        const number z = p[2];
        const number two_pi = number(2.0 * M_PI);
        const number four_pi = number(4.0 * M_PI);
        const number three_pi = number(3.0 * M_PI);

        const number u =
            std::sin(two_pi * x) *
            std::sin(four_pi * y) *
            std::sin(three_pi * z);

        const number ux =
            two_pi * std::cos(two_pi * x) *
            std::sin(four_pi * y) *
            std::sin(three_pi * z);

        const number lambda = number(29.0 * M_PI * M_PI); // -Δu = λ u
        const number k = number(1.0);
        const number beta = number(beta0);

        return (lambda + k + beta) * u + beta * (x - number(1.0)) * ux;
      }
      else
      {
        return number(0.0);
      }
    }

  private:
    double beta0;
  };

  // Dirichlet boundary conditions.
  class FunctionG : public Function<dim>
  {
  public:
    // Constructor.
    FunctionG()
    {
    }

    // Evaluation.
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0;
    }
  };

  // Exact solution.
  class ExactSolution : public Function<dim>
  {
  public:
    // Constructor.
    ExactSolution()
    {
    }

    // Evaluation.
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      if (dim == 2)
        return sin(2.0 * M_PI * p[0]) * sin(4.0 * M_PI * p[1]);
      // 3d
      if (dim == 3)
        return std::sin(2.0 * M_PI * p[0]) *
               std::sin(4.0 * M_PI * p[1]) *
               std::sin(3.0 * M_PI * p[2]);
    }

    // Gradient evaluation.
    // deal.II requires this method to return a Tensor (not a double), i.e. a
    // dim-dimensional vector. In our case, dim = 1, so that the Tensor will in
    // practice contain a single number. Nonetheless, we need to return an
    // object of type Tensor.
    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> result;

      // Points 3 and 4.   for 2d
      if (dim == 2)
      {
        result[0] = 2 * M_PI * cos(2 * M_PI * p[0]) * sin(4 * M_PI * p[1]);
        result[1] = 4 * M_PI * sin(2 * M_PI * p[0]) * cos(4 * M_PI * p[1]);
      }

      // for 3d
      if (dim == 3)
      {

        result[0] = 2.0 * M_PI * std::cos(2.0 * M_PI * p[0]) *
                    std::sin(4.0 * M_PI * p[1]) *
                    std::sin(3.0 * M_PI * p[2]);
        result[1] = 4.0 * M_PI * std::sin(2.0 * M_PI * p[0]) *
                    std::cos(4.0 * M_PI * p[1]) *
                    std::sin(3.0 * M_PI * p[2]);
        result[2] = 3.0 * M_PI * std::sin(2.0 * M_PI * p[0]) *
                    std::sin(4.0 * M_PI * p[1]) *
                    std::cos(3.0 * M_PI * p[2]);
      }
      return result;
    }
  };

  // Constructor.
  Poisson3DParallelMf(const int _N, const unsigned int &r_, const double beta0_in = 0.0)
      : N(_N), r(r_),
        beta0(beta0_in), advection_coefficient(beta0_in), forcing_term(beta0_in),
        mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
        mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
        mesh(MPI_COMM_WORLD), pcout(std::cout, mpi_rank == 0)
  {
  }

  // Initialization.
  void
  setup();

  // System assembly.
  void
  assemble();

  // System solution.
  void
  solve();

  // Output.
  void
  output() const;

  double compute_error(const VectorTools::NormType &norm_type) const;

  double get_memory_consumption() const;

  double get_process_rss_MB()
{
  Utilities::System::MemoryStats stats;
  Utilities::System::get_memory_stats(stats);
  return stats.VmHWM / 1024.0; // MB for this rank
}


  // --- HPC metrics helpers ---
  unsigned int
  get_last_cg_iterations() const
  {
    return last_cg_iterations;
  }

  double
  get_number_of_dofs() const
  {
    return dof_handler.n_dofs();
  }

public:
  // Path to the mesh file.
  const int N;
  // Polynomial degree.
  const unsigned int r;

  
  // Diffusion coefficient.
  DiffusionCoefficient diffusion_coefficient;
  
  // Reaction coefficient.
  ReactionCoefficient reaction_coefficient;
  
  // Advection strength and coefficient
  double beta0;
  AdvectionCoefficient advection_coefficient;
  
  // Forcing term.
  ForcingTerm forcing_term;
  
  // g(x).
  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;
  FunctionG function_g;

  // Triangulation. The parallel::fullydistributed::Triangulation class manages
  // a triangulation that is completely distributed (i.e. each process only
  // knows about the elements it owns and its ghost elements).
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // System matrix.
  TrilinosWrappers::SparseMatrix system_matrix;

  // System right-hand side.
  TrilinosWrappers::MPI::Vector system_rhs;

  // System solution.
  TrilinosWrappers::MPI::Vector solution;

  // Parallel output stream.
  ConditionalOStream pcout;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> constraints;

  // HPC statistics
  unsigned int last_cg_iterations = 0;
  double last_cg_residual = 0.0;
};

#endif