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
  using VectorType = LinearAlgebra::distributed::Vector<Number>; // same as solution/rhs

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
    // Constructor.
    ForcingTerm()
    {
    }

    // Evaluation.
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      // for 2d mesh
      if (dim == 2)
      {
        return (20.0 * M_PI * M_PI + 1.0) * sin(2.0 * M_PI * p[0]) * sin(4.0 * M_PI * p[1]);
      }
      // for 3d mesh
      if (dim == 3)
      {
        return (29.0 * M_PI * M_PI + 1.0) *
               std::sin(2.0 * M_PI * p[0]) *
               std::sin(4.0 * M_PI * p[1]) *
               std::sin(3.0 * M_PI * p[2]);
      }
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

    virtual void compute_diagonal() override {

    };

    void evaluate_coefficient(const DiffusionCoefficient<dim> &diffusion_function,
                              const ReactionCoefficient<dim> &reaction_function)
    {
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

    void initialize(std::shared_ptr<MatrixFree<dim, Number>> mf_ptr)
    {
      this->mf = mf_ptr;
      this->data = mf_ptr;
      // allocate temporaries (MatrixFree sizes things according to dof layout)
      const auto &partitioner = mf_ptr->get_vector_partitioner();
      tmp_dst.reinit(partitioner);
      tmp_src.reinit(partitioner);

      this->mf->initialize_dof_vector(src_ghost, 0); // ghosted
    }

    void set_constraints(const AffineConstraints<Number> &c) { constraints_ptr = &c; }
    // vmult: compute dst = A * src
    void vmult(VectorType &dst, const VectorType &src) const
    {
      //src_ghost = src;
      //src_ghost.update_ghost_values();

      //if (constraints_ptr)
        //constraints_ptr->set_zero(src_ghost);

      dst = 0;
      mf->cell_loop(&MatrixFreeLaplaceOperator::local_apply, this, dst, src);
      //dst.compress(VectorOperation::add); // Communicate additions to ghost entries
      //if (constraints_ptr)
        //constraints_ptr->set_zero(dst);
    }
    //0.0250 1.00 4.0874e-03 2.01 6.1929e-01 0.99 with set_zero
    //0.0250 1.00 4.2253e-03 1.98 6.1764e-01 0.99 without set_zero

    // compute diagonal (optional helper) to do
    void local_compute_diagonal(FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> &fe_eval) const
    {
      /* FEEvaluation<dim, -1, 0, 0, Number> fe_eval(data); // No gradients needed for diagonal

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
       }*/
    }

  private:
    std::shared_ptr<MatrixFree<dim, Number>> mf;
    mutable VectorType tmp_dst, tmp_src;
    const AffineConstraints<Number> *constraints_ptr;
    // DiffusionCoefficient diffusion;
    // ReactionCoefficient reaction;
    Table<2, VectorizedArray<number>> diffusion_coefficient;
    Table<2, VectorizedArray<number>> reaction_coefficient;

    mutable VectorType src_ghost;

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
  Poisson3DParallel(const std::string &mesh_file_name_)
      : mesh_file_name(mesh_file_name_), r(fe_degree), mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)), mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)), mesh(MPI_COMM_WORLD), pcout(std::cout, mpi_rank == 0)
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
};

#endif
