#ifndef __MATRIX_VIEW_H
#define __MATRIX_VIEW_H

#include <vector>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/common/Array.h>
#include <dolfin/log/log.h>

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
               const Array<std::size_t>& rows,
               const Array<std::size_t>& cols)
      : _A(A),
        _rows(rows.size(), const_cast<std::size_t*>(rows.data())),
        _cols(rows.size(), const_cast<std::size_t*>(cols.data()))
    { 
      // TODO: check indices?
    }

    /// Copy constructor
    MatrixView(const MatrixView& mv)
      : _A(mv._A), 
        _rows(mv._rows.size(), const_cast<std::size_t*>(mv._rows.data())),
        _cols(mv._cols.size(), const_cast<std::size_t*>(mv._cols.data()))
    { }

    /// Destructor
    virtual ~MatrixView() { }

    /// Return indices
    void inds(Array<std::size_t>& indices, std::size_t dim) const
    {
      const Array<std::size_t>* _inds;
      switch (dim)
      {
        case 0:
          _inds = &_rows;
        case 1:
          _inds = &_cols;
        default:
          dolfin_error("MatrixView.h",
                       "return indices of MatrixView",
                       "Supplied dim is wrong");
      }
      if (indices.size() != _inds->size())
        dolfin_error("MatrixView.h",
                     "return indices of MatrixView",
                     "Size of supplied indices does not match");
      for (std::size_t i = 0; i < indices.size(); ++i)
        indices[i] = (*_inds)[i];
    }

    /// Return size of given dimension
    virtual std::size_t size(std::size_t dim) const
    {
      switch (dim)
      {
        case 0:
          return _rows.size();
        case 1:
          return _cols.size();
        default:
          dolfin_error("MatrixView.h",
                       "return indices of MatrixView",
                       "Supplied dim is wrong");
      }
    }

    /// Return local ownership range
    virtual std::pair<std::size_t, std::size_t>
      local_range(std::size_t dim) const
    {
      dolfin_assert(_A);
      // TODO: this is wrong!
      return _A->local_range(dim);
    }

    /// Set all entries to zero and keep any sparse structure
    virtual void zero()
    { 
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Finalize assembly of tensor
    virtual void apply(std::string mode)
    { dolfin_assert(_A);
      _A->apply(mode); }

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const
    { 
      //TODO: implement!
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
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Get block of values
    virtual void get(double* block,
                     std::size_t m, const dolfin::la_index* rows,
                     std::size_t n, const dolfin::la_index* cols) const
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Set block of values
    virtual void set(const double* block,
                     std::size_t m, const dolfin::la_index* rows,
                     std::size_t n, const dolfin::la_index* cols)
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Add block of values
    virtual void add(const double* block,
                     std::size_t m, const dolfin::la_index* rows,
                     std::size_t n, const dolfin::la_index* cols)
    {
      // TODO: Dynamic allocation of memory?! Not good!
      std::vector<std::vector<dolfin::la_index> > rowcols;
      rowcols.resize(2);
      rowcols[0].resize(m);
      rowcols[1].resize(n);
      for (std::size_t i = 0; i < m; ++i)
        rowcols[0][i] = _rows[rows[i]];
      for (std::size_t i = 0; i < n; ++i)
        rowcols[1][i] = _cols[cols[i]];
      _A->add(block, rowcols);
    }

    /// Add multiple of given matrix (AXPY operation)
    virtual void axpy(double a, const GenericMatrix& A,
                      bool same_nonzero_pattern)
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Return norm of matrix
    virtual double norm(std::string norm_type) const
    {
      //TODO: implement!
      dolfin_not_implemented();
      return 0.0;
    }

    /// Get non-zero values of given row on local process
    virtual void getrow(std::size_t row, std::vector<std::size_t>& columns,
                        std::vector<double>& values) const
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Set values for given row on local process
    virtual void setrow(std::size_t row,
                        const std::vector<std::size_t>& columns,
                        const std::vector<double>& values)
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Set given rows to zero
    virtual void zero(std::size_t m, const dolfin::la_index* rows)
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Set given rows to identity matrix
    virtual void ident(std::size_t m, const dolfin::la_index* rows)
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Matrix-vector product, y = A^T x. The y vector must either be
    /// zero-sized or have correct size and parallel layout.
    virtual void transpmult(const GenericVector& x, GenericVector& y) const
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Multiply matrix by given number
    virtual const MatrixView& operator*= (double a)
    {
      //TODO: implement!
      dolfin_not_implemented();
      return *this;
    }

    /// Divide matrix by given number
    virtual const MatrixView& operator/= (double a)
    {
      //TODO: implement!
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
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Return true if empty
    virtual bool empty() const
    {
      //TODO: implement!
      dolfin_not_implemented();
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
      //TODO: implement!
      dolfin_not_implemented();
    }

  private:

    std::shared_ptr<GenericMatrix> _A;

    Array<std::size_t> _rows;

    Array<std::size_t> _cols;

  };

}

#endif
