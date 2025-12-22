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
#include <memory>
#include <iomanip>
#include <iostream>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

using namespace dealii;

// Class representing the non-linear diffusion problem.
class Heat
{

public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;
  static constexpr unsigned int fe_degree = 3;

  using NUMBER = double;
  using VectorType = LinearAlgebra::distributed::Vector<NUMBER>; // same as solution/rhs

  enum class PreconditionerType
  {
    None,
    Jacobi
  };

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
  AdvectionCoefficient(const double beta0_in = 0.0)
    : Function<dim>(), beta0(beta0_in)
  {}

  virtual double
  value(const Point<dim> &p,
        const unsigned int component = 0) const override
  {
    if (component == 0)
      return beta0 * (p[0] - 1.0);  // b_x = beta0 (x-1)
    else
      return 0.0;
  }

  virtual void
  vector_value(const Point<dim> &p,
               Vector<double> &values) const override
  {
    for (unsigned int d = 0; d < dim; ++d)
      values[d] = value<NUMBER>(p, d);
  }

  template <typename number>
  number value(const Point<dim, number> &p,
               const unsigned int component = 0) const
  {
    if (component == 0)
      return number(beta0) * (p[0] - number(1.0));
    else
      return number(0.0);
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

  // Function for the forcing term.
  class ForcingTerm : public Function<dim>
{
public:
  explicit ForcingTerm(const double beta0_in = 0.0)
    : Function<dim>(), beta0(beta0_in)
  {}

  virtual double
  value(const Point<dim> &p,
        const unsigned int /*component*/ = 0) const override
  {
    const double t   = this->get_time();
    const double G   = std::sin(numbers::PI * t);
    const double G_t = numbers::PI * std::cos(numbers::PI * t);

    const double k    = 1.0;
    const double beta = beta0;

    // dim == 3 (your Heat::dim)
    const double x = p[0];
    const double y = p[1];
    const double z = p[2];

    const double two_pi   = 2.0 * numbers::PI;
    const double four_pi  = 4.0 * numbers::PI;
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

    // f = U G' + (λ + k + beta) U G + beta (x-1) U_x G
    return U * G_t
         + (lambda + k + beta) * U * G
         + beta * (x - 1.0) * U_x * G;
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
    const double t  = this->get_time();
    const double Gt = std::sin(numbers::PI * t);

    const double x = p[0];
    const double y = p[1];
    const double z = p[2];

    const double u_spatial =
      std::sin(2.0 * numbers::PI * x) *
      std::sin(4.0 * numbers::PI * y) *
      std::sin(3.0 * numbers::PI * z);

    return u_spatial * Gt;
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> &p,
           const unsigned int /*component*/ = 0) const override
  {
    Tensor<1, dim> result;

    const double t  = this->get_time();
    const double Gt = std::sin(numbers::PI * t);

    const double x = p[0];
    const double y = p[1];
    const double z = p[2];

    const double two_pi   = 2.0 * numbers::PI;
    const double four_pi  = 4.0 * numbers::PI;
    const double three_pi = 3.0 * numbers::PI;

    result[0] =
      two_pi * std::cos(two_pi * x) *
      std::sin(four_pi * y) *
      std::sin(three_pi * z) * Gt;

    result[1] =
      four_pi * std::sin(two_pi * x) *
      std::cos(four_pi * y) *
      std::sin(three_pi * z) * Gt;

    result[2] =
      three_pi * std::sin(two_pi * x) *
      std::sin(four_pi * y) *
      std::cos(three_pi * z) * Gt;

    return result;
  }
};


  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  Heat(const int N_,
       const std::string &mesh_file_name_,
       // const unsigned int &r_,
       const double &T_,
       const double &deltat_,
       const double &theta_,
       const PreconditionerType preconditioner_type_,
       const double beta0_ = 0.0)
      : b(beta0_), forcing_term(beta0_), N(N_), mesh_file_name(mesh_file_name_), T(T_), deltat(deltat_), theta(theta_), 
      time(0.0), mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)), 
      mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)), 
      pcout(std::cout, mpi_rank == 0), mesh(MPI_COMM_WORLD),
       mf_operator_lhs(deltat, theta, true), mf_operator_rhs(deltat, theta, false),
        preconditioner_type(preconditioner_type_), beta0(beta0_)

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

  double get_memory_consumption() const;

  double get_process_rss_MB()
{
  Utilities::System::MemoryStats stats;
  Utilities::System::get_memory_stats(stats);
  return stats.VmHWM / 1024.0; // MB for this rank
}


  void set_timer(const std::shared_ptr<TimerOutput> &timer_) { timer = timer_; }

  // High-level performance stats
  unsigned int get_n_time_steps() const { return n_time_steps; }
  unsigned long long get_total_gmres_iterations() const { return total_gmres_iterations; }
  double get_total_linear_solve_time() const { return total_linear_solve_time; }

  unsigned int get_number_of_dofs() const { return dof_handler.n_dofs(); }

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
      this->inverse_diagonal_entries.reset(
          new DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>());
      LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
          this->inverse_diagonal_entries->get_vector();
      this->data->initialize_dof_vector(inverse_diagonal);
      unsigned int dummy = 0;
      this->data->cell_loop(&TimeStepOperator::local_compute_diagonal,
                            this,
                            inverse_diagonal,
                            dummy);

      this->set_constrained_entries_to_one(inverse_diagonal);

      for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i)
      {
        Assert(inverse_diagonal.local_element(i) > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero"));
        inverse_diagonal.local_element(i) =
            1. / inverse_diagonal.local_element(i);
      }
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
      // this->data = mf_ptr;
      MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::initialize(mf_ptr);
    }

    // void set_constraints(const AffineConstraints<number> &c) {}

    void local_compute_diagonal(
        const MatrixFree<dim, number> &data,
        LinearAlgebra::distributed::Vector<number> &dst,
        const unsigned int &,
        const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);
      const NUMBER factor_m = 1.0 / deltat;                    // mass matrix
      const NUMBER factor_s = is_lhs ? theta : -(1.0 - theta); // stifness
      AlignedVector<VectorizedArray<number>> diagonal(phi.dofs_per_cell);

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {

        AssertDimension(diffusion_coefficient.size(0), data.n_cell_batches());
        AssertDimension(diffusion_coefficient.size(1), phi.n_q_points);

        AssertDimension(reaction_coefficient.size(0), data.n_cell_batches());
        AssertDimension(reaction_coefficient.size(1), phi.n_q_points);

        AssertDimension(advection_coefficient.size(0), data.n_cell_batches());
        AssertDimension(advection_coefficient.size(1), phi.n_q_points);

        phi.reinit(cell);
        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
            phi.submit_dof_value(VectorizedArray<number>(), j);
          phi.submit_dof_value(make_vectorized_array<number>(1.), i);

          phi.evaluate(EvaluationFlags::gradients | EvaluationFlags::values);
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const auto u = phi.get_value(q);
            const auto grad_u = phi.get_gradient(q);

            auto &b = advection_coefficient(cell, q);
            const auto flux = b * u;
            // Weak form terms: (1/Δt) u v ± θ (μ ∇u · ∇v - (b · ∇u) v + k u v)
            phi.submit_value(factor_m * u + factor_s * reaction_coefficient(cell, q) * u, q);
            phi.submit_gradient(factor_s * (diffusion_coefficient(cell, q) * grad_u - flux), q);
          }
          phi.integrate(EvaluationFlags::gradients | EvaluationFlags::values);
          diagonal[i] = phi.get_dof_value(i);
        }

        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
          phi.submit_dof_value(diagonal[i], i);
        phi.distribute_local_to_global(dst);
      }
    }

  private:
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
  // const unsigned int r;

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

  std::shared_ptr<TimerOutput> timer; // shared, non-owning from Heat's perspective

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

  PreconditionerType preconditioner_type;
  typename MatrixFree<dim, NUMBER>::AdditionalData additional_data;

    double beta0; // advection strength shared by b and f

private:
  // --- High-level performance counters ---
  unsigned int n_time_steps = 0;
  unsigned long long total_gmres_iterations = 0;
  double total_linear_solve_time = 0.0; // [s]
};

#endif