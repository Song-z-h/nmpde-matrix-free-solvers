#ifndef POISSON_3D_HPP
#define POISSON_3D_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
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

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/base/aligned_vector.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

using namespace dealii;

/**
 * Class managing the differential problem.
 */
class Poisson3DParallelMf
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;
  static constexpr unsigned int fe_degree = 3;
  static constexpr bool use_gmg = false;
  using Number = double;
  using VectorType = LinearAlgebra::distributed::Vector<Number>; // same as solution/rhs

  template <typename VectorType>
  void copy_vector_data(VectorType &dst, const VectorType &src)
  {
    // Explicitly copy locally owned elements index-by-index
    // This bypasses layout/padding mismatches because we only touch valid DoFs.
    const auto &local_indices = src.locally_owned_elements();

    // Ensure dst is zeroed first (crucial for padding areas)
    dst = 0.0;

    for (const auto index : local_indices)
    {
      dst[index] = src[index];
    }

    // Ghosts must be updated after this manual copy
    dst.compress(VectorOperation::insert);
  }
  // Diffusion coefficient.
  // In deal.ii, functions are implemented by deriving the dealii::Function
  // class, which provides an interface for the computation of function values
  // and their derivatives
  template <int dim>
  class DiffusionCoefficient : public Function<dim>
  {
  public:
    // Constructor.
    DiffusionCoefficient()
    {
    }

    // Evaluation.
    virtual double
    value(const Point<dim> &p,
          const unsigned int component = 0) const override
    {
      return value<double>(p, component);
    }

    template <typename number>
    number value(const Point<dim, number> & /*p*/,
                 const unsigned int /*component*/ = 0) const
    {
      return 1.0;
    }
  };

  // Reaction coefficient.
  template <int dim>
  class ReactionCoefficient : public Function<dim>
  {
  public:
    // Constructor.
    ReactionCoefficient()
    {
    }

    // Evaluation.
    virtual double
    value(const Point<dim> &p,
          const unsigned int component = 0) const override
    {
      return value<double>(p, component);
    }

    template <typename number>
    number value(const Point<dim, number> & /*p*/,
                 const unsigned int /*component*/ = 0) const
    {
      return 1.0;
    }
  };

  template <int dim>
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
      return value<Number>(p, component);
    }

    template <typename number>
    number value(const Point<dim, number> &p,
                 const unsigned int component = 0) const
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
      for (unsigned int i = 0; i < dim; ++i)
        values[i] = value<Number>(p, i);
    }

  private:
    double beta0;
  };

  // Forcing term.
  // Forcing term.
class ForcingTerm : public Function<dim>
{
public:
  explicit ForcingTerm(const double beta0_in = 0.0)
    : Function<dim>()
    , beta0(beta0_in)
  {}

  double value(const Point<dim> &p,
               const unsigned int = 0) const override
  {
    return value<double>(p);
  }

  template <typename number>
  number value(const Point<dim, number> &p,
               const unsigned int = 0) const
  {
    // Must be consistent with:
    //  - DiffusionCoefficient (mu = 1.0)
    //  - ReactionCoefficient (k = 1.0)
    //  - AdvectionCoefficient: b = (beta0*(x-1), 0, 0)

    const number x = p[0];

    if constexpr (dim == 2)
    {
      const number y       = p[1];
      const number two_pi  = number(2.0 * M_PI);
      const number four_pi = number(4.0 * M_PI);

      const number u =
        std::sin(two_pi * x) *
        std::sin(four_pi * y);

      const number ux =
        two_pi * std::cos(two_pi * x) *
        std::sin(four_pi * y);

      const number lambda = number(20.0 * M_PI * M_PI); // from -Δu = λ u
      const number k      = number(1.0);
      const number beta   = number(beta0);

      // f = -Δu + div(b u) + k u
      //   = (λ + k + beta) u + beta (x-1) ux
      return (lambda + k + beta) * u
           + beta * (x - number(1.0)) * ux;
    }
    else if constexpr (dim == 3)
    {
      const number y       = p[1];
      const number z       = p[2];
      const number two_pi  = number(2.0 * M_PI);
      const number four_pi = number(4.0 * M_PI);
      const number three_pi= number(3.0 * M_PI);

      const number u =
        std::sin(two_pi * x) *
        std::sin(four_pi * y) *
        std::sin(three_pi * z);

      const number ux =
        two_pi * std::cos(two_pi * x) *
        std::sin(four_pi * y) *
        std::sin(three_pi * z);

      const number lambda = number(29.0 * M_PI * M_PI); // -Δu = λ u
      const number k      = number(1.0);
      const number beta   = number(beta0);

      return (lambda + k + beta) * u
           + beta * (x - number(1.0)) * ux;
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
      // 2d
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

  // MatrixFreeLaplaceOperator.hpp  (minimal, illustrative)
  template <int dim, int fe_degree, typename number>
  class MatrixFreeLaplaceOperator : public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>
  {
  public:
    using value_type = number;

    MatrixFreeLaplaceOperator()
        : MatrixFreeOperators::Base<dim,
                                    LinearAlgebra::distributed::Vector<number>>()
    {
    }
    void clear() override
    {
      MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::
          clear();
    }

    virtual void compute_diagonal() override
    {
      this->inverse_diagonal_entries.reset(
          new DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>());
      LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
          this->inverse_diagonal_entries->get_vector();
      this->data->initialize_dof_vector(inverse_diagonal);
      unsigned int dummy = 0;
      this->data->cell_loop(&MatrixFreeLaplaceOperator::local_compute_diagonal,
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
                              const ReactionCoefficient<dim> &reaction_function,
                              const AdvectionCoefficient<dim> &advection_function)
    {
      if (!this->data)
        return; // Safety check
      const unsigned int n_cells = this->data->n_cell_batches();
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(*this->data);

      diffusion_coefficient.reinit(n_cells, phi.n_q_points);
      reaction_coefficient.reinit(n_cells, phi.n_q_points);
      advection_coefficient.reinit(n_cells, phi.n_q_points);

      for (unsigned int cell = 0; cell < n_cells; ++cell)
      {
        phi.reinit(cell);
        for (const unsigned int q : phi.quadrature_point_indices())
        {
          const auto p = phi.quadrature_point(q);

          diffusion_coefficient(cell, q) =
              diffusion_function.value(phi.quadrature_point(q));
          reaction_coefficient(cell, q) =
              reaction_function.value(phi.quadrature_point(q));

          Tensor<1, dim, VectorizedArray<number>> b_loc;
          for (unsigned int d = 0; d < dim; ++d)
            b_loc[d] = advection_function.value(p, d);

          advection_coefficient(cell, q) = b_loc;
        }
      }
    }

    void initialize(const std::shared_ptr<const MatrixFree<dim, number>> &mf_ptr)
    {
      MatrixFreeOperators::Base<dim,
                                LinearAlgebra::distributed::Vector<number>>::initialize(mf_ptr);
    }

    void initialize(const std::shared_ptr<const MatrixFree<dim, number>> &mf_ptr,
                    const MGConstrainedDoFs &mg_constraints,
                    const unsigned int level)
    {
      MatrixFreeOperators::Base<dim,
                                LinearAlgebra::distributed::Vector<number>>::initialize(mf_ptr, mg_constraints, level);
    }

    void local_compute_diagonal(
        const MatrixFree<dim, number> &data,
        LinearAlgebra::distributed::Vector<number> &dst,
        const unsigned int &,
        const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);
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

            // auto &b = advection_coefficient(cell, q);
            // const auto flux = b * u;
            //  Weak form terms: (1/Δt) u v ± θ (μ ∇u · ∇v - (b · ∇u) v + k u v)
            phi.submit_value(reaction_coefficient(cell, q) * u, q);
            phi.submit_gradient(diffusion_coefficient(cell, q) * grad_u - advection_coefficient(cell, q) * u, q);
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
    // std::shared_ptr<MatrixFree<dim, Number>> mf;
    mutable VectorType tmp_dst, tmp_src;
    // const AffineConstraints<Number> *constraints_ptr;
    //  DiffusionCoefficient diffusion;
    //  ReactionCoefficient reaction;
    Table<2, VectorizedArray<number>> diffusion_coefficient;
    Table<2, VectorizedArray<number>> reaction_coefficient;
    Table<2, Tensor<1, dim, VectorizedArray<number>>> advection_coefficient;

    virtual void apply_add(
        LinearAlgebra::distributed::Vector<number> &dst,
        const LinearAlgebra::distributed::Vector<number> &src) const override
    {
      this->data->cell_loop(&MatrixFreeLaplaceOperator::local_apply, this, dst, src);
    }

    // In Poisson3D_parallel.hpp, inside the MatrixFreeLaplaceOperator class

    void local_apply(const MatrixFree<dim, Number> &data,
                     VectorType &dst,
                     const VectorType &src,
                     const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      // The FEEvaluation helper class provides the logic for evaluating FE
      // functions on multiple cells at once (vectorization).

      // FEEvaluation<dim, fe_degree> fe_eval(data);
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> fe_eval(data);

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {

        AssertDimension(diffusion_coefficient.size(0), data.n_cell_batches());
        AssertDimension(diffusion_coefficient.size(1), fe_eval.n_q_points);

        AssertDimension(reaction_coefficient.size(0), data.n_cell_batches());
        AssertDimension(reaction_coefficient.size(1), fe_eval.n_q_points);

        AssertDimension(advection_coefficient.size(0), data.n_cell_batches());
        AssertDimension(advection_coefficient.size(1), fe_eval.n_q_points);

        fe_eval.reinit(cell);
        fe_eval.read_dof_values(src);
        fe_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        for (const unsigned int q : fe_eval.quadrature_point_indices())
        {

          const auto u = fe_eval.get_value(q);
          const auto grad_u = fe_eval.get_gradient(q);

          const auto diff_flux = diffusion_coefficient(cell, q) * grad_u;
          const auto beta_u = advection_coefficient(cell, q) * u;
          const auto flux = diff_flux - beta_u; // μ∇u - b u

          fe_eval.submit_gradient(flux, q);
          fe_eval.submit_value(reaction_coefficient(cell, q) * u, q);
        }

        // Integrate the submitted values and distribute the results
        // from the local cell calculation into the global destination vector.
        fe_eval.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        fe_eval.distribute_local_to_global(dst);
      }
    }
  };

  // Constructor.
  Poisson3DParallelMf(const int _N, const double beta0_in = 0.0)
      : N(_N),
        r(fe_degree),
        mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
        mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
        beta0(beta0_in),
        advection_coefficient(beta0_in),
        forcing_term(beta0_in),
        mesh(MPI_COMM_WORLD),
        pcout(std::cout, mpi_rank == 0)
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

  // --- HPC metrics helpers ---

  double get_process_rss_MB()
{
  Utilities::System::MemoryStats stats;
  Utilities::System::get_memory_stats(stats);
  return stats.VmHWM / 1024.0; // MB for this rank
}

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

protected:
  // Path to the mesh file.
  // const std::string mesh_file_name;
  const int N;

  // Polynomial degree.
  const unsigned int r;

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Diffusion coefficient.
  // DiffusionCoefficient diffusion_coefficient;

  // Reaction coefficient.
  // ReactionCoefficient reaction_coefficient;
  double beta0;

  DiffusionCoefficient<dim> diffusion_coefficient;
  ReactionCoefficient<dim> reaction_coefficient;
  AdvectionCoefficient<dim> advection_coefficient;

  // Forcing term.
  ForcingTerm forcing_term;

  // g(x).
  FunctionG function_g;

  // Triangulation. The parallel::fullydistributed::Triangulation class manages
  // a triangulation that is completely distributed (i.e. each process only
  // knows about the elements it owns and its ghost elements).
 // parallel::distributed::Triangulation<dim> mesh;
   parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // System matrix.
  // TrilinosWrappers::SparseMatrix system_matrix;

  // System right-hand side.
  // TrilinosWrappers::MPI::Vector system_rhs;

  // System solution.
  // TrilinosWrappers::MPI::Vector solution;

  // Parallel output stream.
  ConditionalOStream pcout;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  // matrix free components

  // used to give apply D- boundary conditions
  AffineConstraints<Number> constraints;

  // using SystemMatrixType =
  //   MatrixFreeLaplaceOperator<dim>;
  // SystemMatrixType system_matrix;

  VectorType solution;
  VectorType system_rhs;

  // Matrix-free pieces
  std::shared_ptr<MatrixFree<dim, Number>> mf_storage;
  MatrixFreeLaplaceOperator<dim, fe_degree, Number> mf_operator;

  typename MatrixFree<dim, Number>::AdditionalData additional_data;

  // --- HPC statistics (for reporting) ---
  unsigned int last_cg_iterations = 0;
  double last_cg_residual = 0.0;

  class DiagonalPreconditioner : public Subscriptor
  {
  public:
    void initialize(const MatrixFree<dim> &matrix_free,
                    MatrixFreeLaplaceOperator<dim, fe_degree, Number> &laplace_op) // Your matrix-free operator class
    {
      laplace_op.compute_diagonal(); // Fills with 1/diagonal elements

      matrix_free.initialize_dof_vector(inverse_diagonal);
      inverse_diagonal = laplace_op.get_matrix_diagonal_inverse()->get_vector();
    }

    void vmult(LinearAlgebra::distributed::Vector<double> &dst,
               const LinearAlgebra::distributed::Vector<double> &src) const
    {
      dst = src;
      dst.scale(inverse_diagonal); // Apply inverse diagonal element-wise
    }

  private:
    LinearAlgebra::distributed::Vector<double> inverse_diagonal;
  };

  // DiagonalPreconditioner preconditioner;

  /*geometric multi grid*/
  using LevelMatrixType = MatrixFreeLaplaceOperator<dim, fe_degree, Number>;
  using SmootherType = PreconditionChebyshev<LevelMatrixType, VectorType>;
  using MGInterfaceType = MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>;

  // 2. MG hierarchy objects
  MGLevelObject<std::shared_ptr<MatrixFree<dim, Number>>> mg_mf_storage;
  MGLevelObject<LevelMatrixType> mg_matrices;
  MGLevelObject<MGInterfaceType> mg_interface_matrices;

  dealii::MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType> mg_smoother;

  // wrappers around level & interface operators
  dealii::mg::Matrix<VectorType> mg_matrix_wrapper;
  dealii::mg::Matrix<VectorType> mg_interface_wrapper;

  MGConstrainedDoFs mg_constrained_dofs;
};

#endif
