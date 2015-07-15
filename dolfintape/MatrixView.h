// Copyright (C) 2015 Jan Blechta
//
// This file is part of dolfin-tape.
//
// dolfin-tape is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// dolfin-tape is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with dolfin-tape. If not, see <http://www.gnu.org/licenses/>.

#ifndef __MATRIX_VIEW_H
#define __MATRIX_VIEW_H

#include <vector>
#include <dolfin/log/log.h>
#include <dolfin/common/types.h>
#include <dolfin/common/Array.h>
#include <dolfin/la/GenericMatrix.h>

namespace dolfin
{

  class GenericVector;
  class TensorLayout;

  /// This class defines a common interface for matrices.

  class MatrixView : public GenericMatrix
  {
  public:

    /// Constructor
    MatrixView(const std::shared_ptr<GenericMatrix> A,
               const std::size_t dim0, const std::size_t dim1,
               const Array<la_index>& rows,
               const Array<la_index>& cols)
      : _A(A), _dim0(dim0), _dim1(dim1),
        _rows(rows.size(), const_cast<la_index*>(rows.data())),
        _cols(rows.size(), const_cast<la_index*>(cols.data()))
    {
      // TODO: check indices?
    }

    /// Copy constructor
    MatrixView(const MatrixView& mv)
      : _A(mv._A), _dim0(mv._dim0), _dim1(mv._dim1),
        _work0(mv._work0.size()), _work1(mv._work1.size()),
        _rows(mv._rows.size(), const_cast<la_index*>(mv._rows.data())),
        _cols(mv._cols.size(), const_cast<la_index*>(mv._cols.data()))
    { }

    /// Destructor
    virtual ~MatrixView() { }

    /// Resize work array for indices of given dim to given size
    void resize_work_array(std::size_t dim, std::size_t size)
    {
      switch (dim)
      {
        case 0:
          _work0.resize(size); return;
        case 1:
          _work1.resize(size); return;
        default:
          dolfin_error("MatrixView.h",
                       "resize work array",
                       "wrong dim (%d)", dim);
      }
    }

    /// Return size of given dimension
    virtual std::size_t size(std::size_t dim) const
    {
      switch (dim)
      {
        case 0:
          return _dim0;
        case 1:
          return _dim1;
        default:
          dolfin_error("MatrixView.h",
                       "return size of MatrixView",
                       "Supplied dim is wrong");
      }
    }

    /// Return local ownership range
    virtual std::pair<std::size_t, std::size_t>
      local_range(std::size_t dim) const
    {
      dolfin_not_implemented();
      return _A->local_range(dim);
    }

    /// Return number of non-zero entries in matrix (collective)
    virtual std::size_t nnz() const
    {
      dolfin_not_implemented();
      return 0;
    }


    /// Set all entries to zero and keep any sparse structure
    virtual void zero()
    {
      dolfin_not_implemented();
    }

    /// Finalize assembly of tensor
    virtual void apply(std::string mode)
    { dolfin_assert(_A);
      _A->apply(mode); }

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const
    {
      dolfin_not_implemented();
      return "Not implemented!";
    }

    //--- Matrix interface ---

    /// Return copy of matrix
    virtual std::shared_ptr<GenericMatrix> copy() const
    { std::shared_ptr<GenericMatrix> cp;
      cp.reset(new MatrixView(*this));
      return cp; }

    /// Initialize vector z to be compatible with the matrix-vector
    /// product y = Ax. In the parallel case, both size and layout are
    /// important.
    ///
    /// *Arguments*
    ///     dim (std::size_t)
    ///         The dimension (axis): dim = 0 --> z = y, dim = 1 --> z = x
    virtual void init_vector(GenericVector& z, std::size_t dim) const
    {
      dolfin_not_implemented();
    }

    /// Get block of values
    virtual void get(double* block,
                     std::size_t m, const dolfin::la_index* rows,
                     std::size_t n, const dolfin::la_index* cols) const
    {
      dolfin_not_implemented();
    }

    /// Set block of values using global indices
    virtual void set(const double* block,
                     std::size_t m, const dolfin::la_index* rows,
                     std::size_t n, const dolfin::la_index* cols)
    {
      dolfin_not_implemented();
    }

    /// Set block of values using local indices
    virtual void set_local(const double* block,
                           std::size_t m, const dolfin::la_index* rows,
                           std::size_t n, const dolfin::la_index* cols)
    {
      dolfin_not_implemented();
    }

    /// Add block of values using global indices
    virtual void add(const double* block,
                     std::size_t m, const dolfin::la_index* rows,
                     std::size_t n, const dolfin::la_index* cols)
    {
      dolfin_not_implemented();
    }

    /// Add block of values using local indices
    virtual void add_local(const double* block,
                           std::size_t m, const dolfin::la_index* rows,
                           std::size_t n, const dolfin::la_index* cols)
    {
      if (m > _work0.size() || n > _work1.size())
        dolfin_error("VectorView.h",
                     "add to parent data vector",
                     "work array(s) too small (%dx%d), required (%dx%d). "
                     "Call resize_work_array(...)",
                     _work0.size(), _work1.size(), m, n);
      for (std::size_t i = 0; i < m; ++i)
        _work0[i] = _rows[rows[i]];
      for (std::size_t i = 0; i < n; ++i)
        _work1[i] = _cols[cols[i]];
      _A->add(block, m, _work0.data(), n, _work1.data());
    }

    /// Add multiple of given matrix (AXPY operation)
    virtual void axpy(double a, const GenericMatrix& A,
                      bool same_nonzero_pattern)
    {
      dolfin_not_implemented();
    }

    /// Return norm of matrix
    virtual double norm(std::string norm_type) const
    {
      dolfin_not_implemented();
      return 0.0;
    }

    /// Get non-zero values of given row (global index) on local process
    virtual void getrow(std::size_t row, std::vector<std::size_t>& columns,
                        std::vector<double>& values) const
    {
      dolfin_not_implemented();
    }

    /// Set values for given row (global index) on local process
    virtual void setrow(std::size_t row,
                        const std::vector<std::size_t>& columns,
                        const std::vector<double>& values)
    {
      dolfin_not_implemented();
    }

    /// Set given rows (global row indices) to zero
    virtual void zero(std::size_t m, const dolfin::la_index* rows)
    {
      dolfin_not_implemented();
    }

    /// Set given rows (local row indices) to zero
    virtual void zero_local(std::size_t m, const dolfin::la_index* rows)
    {
      dolfin_not_implemented();
    }

    /// Set given rows (global row indices) to identity matrix
    virtual void ident(std::size_t m, const dolfin::la_index* rows)
    {
      dolfin_not_implemented();
    }

    /// Set given rows (local row indices) to identity matrix
    virtual void ident_local(std::size_t m, const dolfin::la_index* rows)
    {
      dolfin_not_implemented();
    }

    /// Matrix-vector product, y = A^T x. The y vector must either be
    /// zero-sized or have correct size and parallel layout.
    virtual void transpmult(const GenericVector& x, GenericVector& y) const
    {
      dolfin_not_implemented();
    }

    /// Set diagonal of a matrix
    virtual void set_diagonal(const GenericVector& x)
    {
      dolfin_not_implemented();
    }

    /// Multiply matrix by given number
    virtual const MatrixView& operator*= (double a)
    {
      dolfin_not_implemented();
      return *this;
    }

    /// Divide matrix by given number
    virtual const MatrixView& operator/= (double a)
    {
      dolfin_not_implemented();
      return *this;
    }

    /// Assignment operator
    // TODO: Shouldn't be disabled?!
    virtual const GenericMatrix& operator= (const GenericMatrix& A)
    {
      *this = as_type<const MatrixView>(A);
      return *this;
    }

  private:

    /// Assignment operator
    virtual const MatrixView& operator= (const MatrixView& A) { }

  public:

    //--- Tensor interface ---

    /// Initialize zero tensor using tensor layout
    virtual void init(const TensorLayout& tensor_layout)
    {
      dolfin_not_implemented();
    }

    /// Return true if empty
    virtual bool empty() const
    {
      // Non-empty from construction time
      return false;
    }

    /// Return MPI communicator
    virtual MPI_Comm mpi_comm() const
    { dolfin_assert(_A);
      return _A->mpi_comm();}

    /// Return linear algebra backend factory
    virtual GenericLinearAlgebraFactory& factory() const
    { dolfin_not_implemented();
      return DefaultFactory::factory(); }

    //--- LinearOperator interface ---

    /// Compute matrix-vector product y = Ax
    virtual void mult(const GenericVector& x, GenericVector& y) const
    {
      dolfin_not_implemented();
    }

  private:

    std::shared_ptr<GenericMatrix> _A;

    std::size_t _dim0, _dim1;

    Array<la_index> _rows, _cols;

    std::vector<la_index> _work0, _work1;

  };

}

#endif
