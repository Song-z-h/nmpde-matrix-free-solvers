#ifndef POISSON_3D_HPP
#define POISSON_3D_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
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

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace dealii;

/**
 * Class managing the differential problem.
 */
class Poisson3DParallel
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 2;
  static constexpr unsigned int fe_degree = 1;
  using Number = double;
  using VectorType = LinearAlgebra::distributed::Vector<Number>; // same as your solution/rhs

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

  // Forcing term.
  class ForcingTerm : public Function<dim>
  {
  public:
    // Constructor.
    ForcingTerm()
    {
    }

    // Evaluation.
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return (20.0 * M_PI * M_PI + 1.0) * sin(2.0 * M_PI * p[0]) * sin(4.0 * M_PI * p[1]);
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
      return sin(2.0 * M_PI * p[0]) * sin(4.0 * M_PI * p[1]);
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
      result[0] = 2 * M_PI * cos(2 * M_PI * p[0]) * sin(4 * M_PI * p[1]);

      result[1] = 4 * M_PI * sin(2 * M_PI * p[0]) * cos(4 * M_PI * p[1]);

      return result;
    }
  };

  // MatrixFreeLaplaceOperator.hpp  (minimal, illustrative)
  template <int dim, int fe_degree, typename number>
  class MatrixFreeLaplaceOperator : public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>
  {
  public:
    using value_type = number;

    MatrixFreeLaplaceOperator() {};

    void clear() override
    {
      MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::
          clear();
    }

    virtual void compute_diagonal() override {

    };

    void initialize(std::shared_ptr<MatrixFree<dim, Number>> mf_ptr)
    {
      this->mf = mf_ptr;

      // allocate temporaries (MatrixFree sizes things according to dof layout)
      const auto &partitioner = mf_ptr->get_vector_partitioner();
      tmp_dst.reinit(partitioner);
      tmp_src.reinit(partitioner);
    }

    // Let the operator know coefficients (copied by value here)
    void set_diffusion(const DiffusionCoefficient &diff)
    {
      diffusion = diff;
    }
    void set_reaction(const ReactionCoefficient &react)
    {
      reaction = react;
    }
    void set_constraints(const AffineConstraints<Number> &c) { constraints_ptr = &c; }

    // vmult: compute dst = A * src
    void vmult(VectorType &dst, const VectorType &src) const
    {
      dst = 0;
      VectorType src_copy = src;
      if (constraints_ptr != nullptr)
    constraints_ptr->set_zero(src_copy);
      // MatrixFree::cell_loop will call local_apply for ranges of cells.
      /*mf->cell_loop(&MatrixFreeLaplaceOperator::local_apply,
                    const_cast<MatrixFreeLaplaceOperator *>(this),
                    dst, src);*/
      mf->cell_loop(&MatrixFreeLaplaceOperator::local_apply, this, dst, src);

      // ensure constrained DoFs remain correct (if required)
      if (this->constraints_ptr != nullptr)
        this->constraints_ptr->distribute(dst);
    }

    // compute diagonal (optional helper)
    // In MatrixFreeLaplaceOperator (header):
    void local_compute_diagonal(const MatrixFree<dim, Number> &data,
                                VectorType &dst,
                                const std::pair<unsigned int, unsigned int> &range) const
    {
      FEEvaluation<dim, -1, 0, 0, Number> fe_eval(data); // No gradients needed for diagonal

      for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        fe_eval.reinit(cell);

        // For diagonal: evaluate operator on unit DoF vectors (MatrixFreeTools handles looping over DoFs)
        // Simplified: reuse local_apply logic but extract diagonal (see deal.II examples for full impl)
        // Stub: zero for now, or implement basis evaluation
        for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
        {
          // Placeholder: submit identity-like for diagonal ~1 (adjust for actual op)
          fe_eval.submit_value(1.0, q); // Reaction term dominates diagonal
        }
        fe_eval.integrate(EvaluationFlags::values);
        fe_eval.distribute_local_to_global(dst);
      }
    }

  private:
    std::shared_ptr<MatrixFree<dim, Number>> mf;
    mutable VectorType tmp_dst, tmp_src;
    const AffineConstraints<Number> *constraints_ptr;
    DiffusionCoefficient diffusion;
    ReactionCoefficient reaction;

    virtual void apply_add(
        LinearAlgebra::distributed::Vector<number> &dst,
        const LinearAlgebra::distributed::Vector<number> &src) const override
    {
      this->data->cell_loop(&MatrixFreeLaplaceOperator::local_apply, this, dst, src);
    };

    // In Poisson3D_parallel.hpp, inside the MatrixFreeLaplaceOperator class

    void local_apply(const MatrixFree<dim, Number> &data,
                     VectorType &dst,
                     const VectorType &src,
                     const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      // The FEEvaluation helper class provides the logic for evaluating FE
      // functions on multiple cells at once (vectorization).
     
      FEEvaluation<dim, fe_degree> fe_eval(data);

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        fe_eval.reinit(cell);
        fe_eval.read_dof_values(src);
        fe_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
        {
          // Since the diffusion and reaction coefficients are constant (1.0),
          // we can create a vectorized constant directly.
          const Point<dim, VectorizedArray<Number>> p_vect = fe_eval.quadrature_point(q);
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

          // Submit the contributions for the diffusion and reaction terms.
          fe_eval.submit_gradient(diff_flux, q);
          fe_eval.submit_value(R * u_val, q);
        }

        // Integrate the submitted values and distribute the results
        // from the local cell calculation into the global destination vector.
        fe_eval.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        fe_eval.distribute_local_to_global(dst);
      }
    }

    // local_compute_diagonal: similar to local_apply but applied to unit vectors.
    static void local_compute_diagonal(const MatrixFree<dim, double> &data,
                                       VectorType &dst,
                                       const VectorType &src,
                                       const std::pair<unsigned int, unsigned int> &range,
                                       MatrixFreeLaplaceOperator *op)
    {

      // The default MatrixFreeTools::compute_diagonal will call this with src being
      // a unit vector over local DoFs. The simplest pattern is to reuse local_apply
      // semantics: apply operator to unit basis columns and extract diagonal entries.
      // We can implement a simplified variant by asking FEEvaluation to read_dof_values(src)
      // and then pick the i-th entry result. But MatrixFreeTools orchestrates the unit-vector pattern;
      // here we implement same structure as local_apply, then after distribute_local_to_global,
      // we pick the appropriate diagonal entry from dst (MatrixFreeTools expects this).
    }
  };

  // Constructor.
  Poisson3DParallel(const std::string &mesh_file_name_, const unsigned int &r_)
      : mesh_file_name(mesh_file_name_), r(r_), mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)), mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)), mesh(MPI_COMM_WORLD), pcout(std::cout, mpi_rank == 0)
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

protected:
  // Path to the mesh file.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Diffusion coefficient.
  DiffusionCoefficient diffusion_coefficient;

  // Reaction coefficient.
  ReactionCoefficient reaction_coefficient;

  // Forcing term.
  ForcingTerm forcing_term;

  // g(x).
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
  // TrilinosWrappers::SparseMatrix system_matrix;

  // System right-hand side.
  // TrilinosWrappers::MPI::Vector system_rhs;

  // System solution.
  // TrilinosWrappers::MPI::Vector solution;

  // Parallel output stream.
  ConditionalOStream pcout;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

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
  MatrixFreeLaplaceOperator<dim, fe_degree, double> mf_operator;

  typename MatrixFree<dim, Number>::AdditionalData additional_data;
};

#endif
