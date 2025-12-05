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

  // Forcing term.
  class ForcingTerm : public Function<dim>
{
public:
  ForcingTerm() = default;

  double value(const Point<dim> &p,
               const unsigned int = 0) const override
  {
    return value<double>(p);
  }

  template <typename number>
  number value(const Point<dim, number> &p,
               const unsigned int = 0) const
  {
    if constexpr (dim == 2)
      return (20.0 * M_PI * M_PI + 1.0) *
             std::sin(2.0 * M_PI * p[0]) *
             std::sin(4.0 * M_PI * p[1]);
    else if constexpr (dim == 3)
      return (29.0 * M_PI * M_PI + 1.0) *
             std::sin(2.0 * M_PI * p[0]) *
             std::sin(4.0 * M_PI * p[1]) *
             std::sin(3.0 * M_PI * p[2]);
    else
      return number(0.0);
  }
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
      // this->inverse_diagonal_entries.reset(new DiagonalMatrix<LinearAlgebra::distributed::Vector<number>());
      // LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
      //   this->inverse_diagonal_entries->get_vector();

      LinearAlgebra::distributed::Vector<number> diagonal;
      this->data->initialize_dof_vector(diagonal); // partitioner-compatible init
      diagonal = 0.0;

      MatrixFreeTools::compute_diagonal(*this->data,
                                        diagonal,
                                        &MatrixFreeLaplaceOperator::local_compute_diagonal,
                                        this);

      this->set_constrained_entries_to_one(diagonal);



      // --- DEBUG: diagonal statistics (min/max over all processes) ---
  const unsigned int local_size = diagonal.local_size();
  double local_min = std::numeric_limits<double>::max();
  double local_max = -std::numeric_limits<double>::max();

  for (unsigned int i = 0; i < local_size; ++i)
  {
    const double v = diagonal.local_element(i);
    if (v < local_min) local_min = v;
    if (v > local_max) local_max = v;
  }

  const MPI_Comm comm = diagonal.get_mpi_communicator();
  const double global_min = Utilities::MPI::min(local_min, comm);
  const double global_max = Utilities::MPI::max(local_max, comm);

  if (Utilities::MPI::this_mpi_process(comm) == 0)
    std::cout << "[DIAG] min(diag) = " << global_min
              << ", max(diag) = " << global_max << std::endl;
  // --- END DEBUG ---   





      this->inverse_diagonal_entries.reset(
          new DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>(diagonal));

      LinearAlgebra::distributed::Vector<number> &inv_vec =
          this->inverse_diagonal_entries->get_vector();

      for (unsigned int i = 0; i < inv_vec.local_size(); ++i)
      {
        const number v = inv_vec.local_element(i);
        // safeguard: if diagonal entry extremely small, clamp to avoid blowup
        if (std::abs(v) < 1e-16)
          inv_vec.local_element(i) = static_cast<number>(1.0);
        else
          inv_vec.local_element(i) = static_cast<number>(1.0) / v;
      }
      inv_vec.compress(VectorOperation::insert);
    };

    void evaluate_coefficient(const DiffusionCoefficient<dim> &diffusion_function,
                              const ReactionCoefficient<dim> &reaction_function)
    {
      if (!this->data)
        return; // Safety check
      const unsigned int n_cells = this->data->n_cell_batches();
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(*this->data);

      diffusion_coefficient.reinit(n_cells, phi.n_q_points);
      reaction_coefficient.reinit(n_cells, phi.n_q_points);
      for (unsigned int cell = 0; cell < n_cells; ++cell)
      {
        phi.reinit(cell);
        for (const unsigned int q : phi.quadrature_point_indices())
        {
          diffusion_coefficient(cell, q) =
              diffusion_function.value(phi.quadrature_point(q));
          reaction_coefficient(cell, q) =
              reaction_function.value(phi.quadrature_point(q));
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

    void local_compute_diagonal(FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> &fe) const
    {
      const unsigned int cell = fe.get_current_cell_index();

      //fe.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
      for (unsigned int q : fe.quadrature_point_indices())
      {
        fe.submit_gradient(diffusion_coefficient(cell, q) * fe.get_gradient(q), q);
        fe.submit_value(reaction_coefficient(cell, q) * fe.get_value(q), q);
      }

      fe.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    }

  private:
    // std::shared_ptr<MatrixFree<dim, Number>> mf;
    mutable VectorType tmp_dst, tmp_src;
    // const AffineConstraints<Number> *constraints_ptr;
    //  DiffusionCoefficient diffusion;
    //  ReactionCoefficient reaction;
    Table<2, VectorizedArray<number>> diffusion_coefficient;
    Table<2, VectorizedArray<number>> reaction_coefficient;

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

        fe_eval.reinit(cell);
        fe_eval.read_dof_values(src);
        fe_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        for (const unsigned int q : fe_eval.quadrature_point_indices())
        {
          // Since the diffusion and reaction coefficients are constant (1.0),
          // we can create a vectorized constant directly.
          /*const Point<dim, VectorizedArray<Number>> p_vect = fe_eval.quadrature_point(q);
          VectorizedArray<Number> D = 0.0;
          for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
          {
            Point<dim> p;
            for (unsigned int d = 0; d < dim; ++d)
              p[d] = p_vect[d][v];
            D[v] = diffusion.value(p);
          }

          VectorizedArray<Number> R = 0.0;
          for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
          {
            Point<dim> p;
            for (unsigned int d = 0; d < dim; ++d)
              p[d] = p_vect[d][v];
            R[v] = reaction.value(p);
          }
          // Get the solution gradient and value at the quadrature point.
          // These are returned as vectorized types.
          const Tensor<1, dim, VectorizedArray<Number>> grad_u = fe_eval.get_gradient(q);
          const VectorizedArray<Number> u_val = fe_eval.get_value(q);

          // All subsequent calculations use vectorized arithmetic.
          // The result diff_flux is also a vectorized tensor.
          Tensor<1, dim, VectorizedArray<Number>> diff_flux;
          for (unsigned int d = 0; d < dim; ++d)
            diff_flux[d] = D * grad_u[d];
          */
          // Submit the contributions for the diffusion and reaction terms.
          fe_eval.submit_gradient(diffusion_coefficient(cell, q) * fe_eval.get_gradient(q), q);
          fe_eval.submit_value(reaction_coefficient(cell, q) * fe_eval.get_value(q), q);
        }

        // Integrate the submitted values and distribute the results
        // from the local cell calculation into the global destination vector.
        fe_eval.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        fe_eval.distribute_local_to_global(dst);
      }
    }
  };

  // Constructor.
  Poisson3DParallelMf(const int _N)
      : N(_N),
        r(fe_degree),
        mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
        mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
        /*mesh(MPI_COMM_WORLD)*/ mesh(MPI_COMM_WORLD,
                                      dealii::Triangulation<dim>::limit_level_difference_at_vertices,
                                      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
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

  DiffusionCoefficient<dim> diffusion_coefficient;
  ReactionCoefficient<dim> reaction_coefficient;

  // Forcing term.
  ForcingTerm forcing_term;

  // g(x).
  FunctionG function_g;

  // Triangulation. The parallel::fullydistributed::Triangulation class manages
  // a triangulation that is completely distributed (i.e. each process only
  // knows about the elements it owns and its ghost elements).
  parallel::distributed::Triangulation<dim> mesh;
  // parallel::fullydistributed::Triangulation<dim> mesh;

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
  double       last_cg_residual   = 0.0;

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
  using MGTransferType = MGTransferMatrixFree<dim, Number>;
  using MGInterfaceType = MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>;

  // 2. MG hierarchy objects
  MGLevelObject<std::shared_ptr<MatrixFree<dim, Number>>> mg_mf_storage;
  MGLevelObject<LevelMatrixType> mg_matrices;
  MGLevelObject<MGInterfaceType> mg_interface_matrices;

  MGTransferType mg_transfer;
  dealii::MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType> mg_smoother;

  // wrappers around level & interface operators
  dealii::mg::Matrix<VectorType> mg_matrix_wrapper;
  dealii::mg::Matrix<VectorType> mg_interface_wrapper;

  MGConstrainedDoFs mg_constrained_dofs;

  class ScaledPreconditioner : public Subscriptor
{
public:
  ScaledPreconditioner(const PreconditionMG<dim, VectorType, MGTransferType> &mg_preconditioner,
                       const double                                          alpha)
    : mg_preconditioner(mg_preconditioner)
    , alpha(alpha)
  {}

  void vmult(VectorType &dst, const VectorType &src) const
  {
    mg_preconditioner.vmult(dst, src);
    dst *= alpha;  // scale by alpha
  }

private:
  const PreconditionMG<dim, VectorType, MGTransferType> &mg_preconditioner;
  const double                                           alpha;
};

};

#endif
