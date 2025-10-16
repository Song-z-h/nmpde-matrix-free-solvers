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

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/lac/precondition.h>

#include <fstream>
#include <iostream>

using namespace dealii;

// Class representing the non-linear diffusion problem.
class Heat
{

public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 1;
  static constexpr unsigned int fe_degree = 2;

  using NUMBER = double;
  using VectorType = LinearAlgebra::distributed::Vector<NUMBER>; // same as solution/rhs

  // Function for the epsilon coefficient.
  template <int dim>
  class DiffusionCoefficient : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int component = 0) const override
    {
      return value<NUMBER>(p, component);
    }

    template <typename number>
    number value(const Point<dim, number> & /*p*/,
                 const unsigned int /*component*/ = 0) const
    {
      return 1.0;
    }
  };
  // coeficient for transport advection
  template <int dim>
  class ReactionCoefficient : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int component = 0) const override
    {
      return value<NUMBER>(p, component);
    }

    template <typename number>
    number value(const Point<dim, number> & /*p*/,
                 const unsigned int /*component*/ = 0) const
    {
      return 1.0;
    }
  };

  // Function for the B coefficient.

  template <int dim>
  class AdvectionCoefficient : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int component = 0) const override
    {
      return value<NUMBER>(p, component);
    }

    template <typename number>
    number value(const Point<dim, number> &p,
                 const unsigned int component = 0) const
    {
      if (component == 0)
        return p[0] - 1.0;
      else
        return 0.0;
    }
    virtual void
    vector_value(const Point<dim> &p,
                 Vector<double> &values) const override
    {
      for (unsigned int i = 1; i < dim; ++i)
        values[i] = value<NUMBER>(p, i);
    }
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

  // Function for the forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      const double pi2 = M_PI / 2.0;
      const double x = p[0];
      const double t = get_time();
      return pi2 * sin(pi2 * x) * cos(pi2 * t) + (pi2 * pi2 + 2.0) * sin(pi2 * x) * sin(pi2 * t) + pi2 * (x - 1.0) * cos(pi2 * x) * sin(pi2 * t);
    }
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
    // Constructor.
    ExactSolution()
    {
    }
    // Evaluation.
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      const double pi2 = M_PI / 2.0;
      const double x = p[0];
      return sin(pi2 * x) * sin(pi2 * get_time());
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

      // Points 3 and 4.
      const double pi2 = M_PI / 2.0;
      const double x = p[0];
      result[0] = pi2 * cos(pi2 * x) * sin(pi2 * get_time());

      return result;
    }

    // static constexpr double A = -4.0 / 15.0 * std::pow(0.5, 2.5);
  };

  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  Heat(const int N_,
       const std::string &mesh_file_name_,
       const unsigned int &r_,
       const double &T_,
       const double &deltat_,
       const double &theta_)
      : N(N_), mesh_file_name(mesh_file_name_), r(r_), T(T_), deltat(deltat_), theta(theta_), time(0.0), mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)), mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)), pcout(std::cout, mpi_rank == 0), mesh(MPI_COMM_WORLD), mf_operator_lhs(deltat, theta, true), mf_operator_rhs(deltat, theta, false)
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

  template <int dim, int fe_degree, typename number>
  class TimeStepOperator : public MatrixFreeOperators::Base<dim, VectorType>
  {
  public:
    TimeStepOperator()
        : MatrixFreeOperators::Base<dim, VectorType>() {}

    TimeStepOperator(double deltat_, double theta_, bool is_lhs_)
        : deltat(deltat_), theta(theta_), is_lhs(is_lhs_) {}

    // must implement
    virtual void compute_diagonal() override
    {
    }

    void evaluate_coefficient(const DiffusionCoefficient<dim> &diffusion_function,
                              const AdvectionCoefficient<dim> &advection_function,
                              const ReactionCoefficient<dim> &reaction_function)
    {
      const unsigned int n_cells = this->data->n_cell_batches();
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, NUMBER> phi(*this->data);

      diffusion_coefficient.reinit(n_cells, phi.n_q_points);
      advection_coefficient.reinit(n_cells, phi.n_q_points);
      reaction_coefficient.reinit(n_cells, phi.n_q_points);

      for (unsigned int cell = 0; cell < n_cells; ++cell)
      {
        phi.reinit(cell);
        for (const unsigned int q : phi.quadrature_point_indices())
        {
          diffusion_coefficient(cell, q) =
              diffusion_function.value(phi.quadrature_point(q));
          // advection_coefficient(cell, q) =
          //   advection_function.value(phi.quadrature_point(q));
          reaction_coefficient(cell, q) =
              reaction_function.value(phi.quadrature_point(q));
          Tensor<1, dim, VectorizedArray<number>> b_loc;

          for (unsigned int d = 0; d < dim; ++d)
            b_loc[d] = advection_function.value(phi.quadrature_point(q), d);
          advection_coefficient(cell, q) = b_loc;
        }
      }
    }

    void initialize(std::shared_ptr<MatrixFree<dim, number>> mf_ptr)
    {
      this->data = mf_ptr;
    }

    //void set_constraints(const AffineConstraints<number> &c) {}

    void vmult(VectorType &dst, const VectorType &src) const
    {
      // src_ghost = src;
      dst = 0;
      this->data->cell_loop(&TimeStepOperator::local_apply, this, dst, src);
    }

    void vmult_add(VectorType &dst, const VectorType &src) const
    {
      this->apply_add(dst, src);
    }

    void local_compute_diagonal(FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> &fe_eval) const
    {
    }

  private:
    mutable VectorType tmp_dst, tmp_src;
    //const AffineConstraints<number> *constraints_ptr;

    Table<2, VectorizedArray<number>> diffusion_coefficient;
    Table<2, Tensor<1, dim, VectorizedArray<number>>> advection_coefficient;
    // Table<2, VectorizedArray<number>> advection_coefficient;
    Table<2, VectorizedArray<number>> reaction_coefficient;

    const double deltat;
    const double theta;
    const bool is_lhs;

    mutable VectorType src_ghost;

    virtual void apply_add(
        VectorType &dst, const VectorType &src) const override
    {
      this->data->cell_loop(&TimeStepOperator::local_apply, this, dst, src, false);
    }
    void local_apply(const MatrixFree<dim, number> &data,
                     VectorType &dst,
                     const VectorType &src,
                     const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);
      const NUMBER factor_m = 1.0 / deltat;                    // mass matrix
      const NUMBER factor_s = is_lhs ? theta : -(1.0 - theta); // stifness
      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {

        AssertDimension(diffusion_coefficient.size(0), data.n_cell_batches());
        AssertDimension(diffusion_coefficient.size(1), phi.n_q_points);

        AssertDimension(reaction_coefficient.size(0), data.n_cell_batches());
        AssertDimension(reaction_coefficient.size(1), phi.n_q_points);

        AssertDimension(advection_coefficient.size(0), data.n_cell_batches());
        AssertDimension(advection_coefficient.size(1), phi.n_q_points);

        phi.reinit(cell);
        phi.read_dof_values(src);
        phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        for (const unsigned int q : phi.quadrature_point_indices())
        {
          const auto u = phi.get_value(q);
          const auto grad_u = phi.get_gradient(q);

          auto &b = advection_coefficient(cell, q);
          const auto flux = b * u;
          // Weak form terms: (1/Δt) u v ± θ (μ ∇u · ∇v - (b · ∇u) v + k u v)
          phi.submit_value(factor_m * u + factor_s * reaction_coefficient(cell, q) * u, q);
          phi.submit_gradient(factor_s * (diffusion_coefficient(cell, q) * grad_u - flux), q);

          // Advection term: -θ (b · ∇u) v
          // For dim=1, advection_coefficient(cell, q) is b[0] = x-1
        }

        phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        phi.distribute_local_to_global(dst);
      }
    }
  };

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
  DiffusionCoefficient<dim> mu;

  AdvectionCoefficient<dim> b;
  // k coefficient
  ReactionCoefficient<dim> k;

  // exact solution
  ExactSolution exact_solution;

  // coef for dirichlet
  FunctionBeta beta;
  FunctionAlpha alpha;

  // stabilization term
  // FunctionTau tau;

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
  // TrilinosWrappers::SparseMatrix mass_matrix;

  // Stiffness matrix A.
  // TrilinosWrappers::SparseMatrix stiffness_matrix;

  // Matrix on the left-hand side (M / deltat + theta A).
  // TrilinosWrappers::SparseMatrix lhs_matrix;

  // Matrix on the right-hand side (M / deltat - (1 - theta) A).
  // TrilinosWrappers::SparseMatrix rhs_matrix;

  // Right-hand side vector in the linear system.
  // TrilinosWrappers::MPI::Vector system_rhs;
  VectorType system_rhs;

  // System solution (without ghost elements).
  // TrilinosWrappers::MPI::Vector solution_owned;
  VectorType solution_owned;

  // System solution (including ghost elements).
  // TrilinosWrappers::MPI::Vector solution;
  VectorType solution;

  // Boundary values map (added for fix).
  std::map<types::global_dof_index, double> boundary_values;

  AffineConstraints<NUMBER> constraints;

  std::shared_ptr<MatrixFree<dim, NUMBER>> mf_storage;
  TimeStepOperator<dim, fe_degree, NUMBER> mf_operator_lhs;
  TimeStepOperator<dim, fe_degree, NUMBER> mf_operator_rhs;

  typename MatrixFree<dim, NUMBER>::AdditionalData additional_data;
};

#endif