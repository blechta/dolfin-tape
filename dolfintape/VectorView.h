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

#ifndef __VECTOR_VIEW_H
#define __VECTOR_VIEW_H

#include <vector>
#include <dolfin/log/log.h>
#include <dolfin/common/types.h>
#include <dolfin/common/Array.h>
#include <dolfin/common/utils.h>
#include <dolfin/la/TensorLayout.h>
#include <dolfin/la/GenericVector.h>

namespace dolfin
{
  template<typename T> class Array;

  /// This class defines a common interface for vectors.

  class VectorView : public GenericVector
  {
  public:

    /// Constructor
    VectorView(const std::shared_ptr<GenericVector> x,
               const std::size_t dim,
               const Array<la_index>& inds)
      : _x(x), _dim(dim),
        _inds(inds.size(), const_cast<la_index*>(inds.data()))
    {
      // TODO: check indices?
    }

    /// Copy constructor
    VectorView(const VectorView& vv)
      : _x(vv._x), _dim(vv._dim), _work(vv._work.size()),
        _inds(vv._inds.size(), const_cast<la_index*>(vv._inds.data()))
    { }

    /// Destructor
    virtual ~VectorView() {}

    /// Resize work array for indices of given dim to given size
    void resize_work_array(std::size_t dim, std::size_t size)
    { dolfin_assert(dim==0); _work.resize(size); }

    /// Adds values of itself to supplied vector. User is responsible of
    /// providing compatible vector; otherwise result is undefined.
    void add_to_vector(std::shared_ptr<GenericVector> x) const
    {
      // Get and check size of supplied vector
      std::size_t n = x->local_size();
      if (n > _inds.size())
        dolfin_error("VectorView.h",
                     "add to another vector",
                     "Dimension of supplied vector (%d) is larger than "
                     "indexing array (%d)", n, _inds.size());

      // Intermediate array with zeros
      Array<double> array(n);
      memset(array.data(), 0, n*sizeof(double));

      // Obtain first n values from self
      _x->get(array.data(), n, _inds.data());

      // Add them to supplied vector
      x->add_local(array);
    }

    //--- Implementation of the GenericTensor interface ---

    /// Set all entries to zero and keep any sparse structure
    virtual void zero()
    {
      dolfin_not_implemented();
    }

    /// Finalize assembly of tensor
    virtual void apply(std::string mode)
    { dolfin_assert(_x);
      _x->apply(mode); }

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const
    {
      dolfin_not_implemented();
      return "Not implemented!";
    }

    //--- Vector interface ---

    /// Return copy of vector
    virtual std::shared_ptr<GenericVector> copy() const
    { std::shared_ptr<GenericVector> cp;
      cp.reset(new VectorView(*this));
      return cp; }

    /// Initialize vector to global size N
    virtual void init(std::size_t N)
    {
      dolfin_not_implemented();
    }

    /// Intitialize vector with given local ownership range
    virtual void init(std::pair<std::size_t, std::size_t> range)
    {
      dolfin_not_implemented();
    }

    /// Initialise vector with given ownership range and with ghost
    /// values
    virtual void init(std::pair<std::size_t, std::size_t> range,
                      const std::vector<std::size_t>& local_to_global_map,
                      const std::vector<la_index>& ghost_indices)
    {
      dolfin_not_implemented();
    }

    /// Return global size of vector
    virtual std::size_t size() const
    { return _dim; }

    /// Return local size of vector
    virtual std::size_t local_size() const
    {
      dolfin_not_implemented();
    }

    /// Return local ownership range of a vector
    virtual std::pair<std::int64_t, std::int64_t> local_range() const
    {
      dolfin_not_implemented();
    }

    /// Determine whether global vector index is owned by this process
    virtual bool owns_index(std::size_t i) const
    {
      dolfin_not_implemented();
    }

    /// Get block of values using global indices (values must all live
    /// on the local process, ghosts cannot be accessed)
    virtual void get(double* block, std::size_t m,
                     const dolfin::la_index* rows) const
    {
      dolfin_not_implemented();
    }

    /// Get block of values using local indices (values must all live
    /// on the local process, ghost are accessible)
    virtual void get_local(double* block, std::size_t m,
                           const dolfin::la_index* rows) const
    {
      dolfin_not_implemented();
    }

    /// Set block of values using global indices
    virtual void set(const double* block, std::size_t m,
                     const dolfin::la_index* rows)
    {
      dolfin_not_implemented();
    }

    /// Set block of values using local indices
    virtual void set_local(const double* block, std::size_t m,
                           const dolfin::la_index* rows)
    {
      dolfin_not_implemented();
    }

    /// Add block of values using global indices
    virtual void add(const double* block, std::size_t m,
                     const dolfin::la_index* rows)
    {
      dolfin_not_implemented();
    }

    /// Add block of values using local indices
    virtual void add_local(const double* block, std::size_t m,
                           const dolfin::la_index* rows)
    {
      if (m > _work.size())
        dolfin_error("VectorView.h",
                     "add to parent data vector",
                     "work array too small (%d), required (%d). "
                     "Call resize_work_array(0, %d)", _work.size(), m, m);
      for (std::size_t i = 0; i < m; ++i)
        _work[i] = _inds[rows[i]];
      _x->add(block, m, _work.data());
    }

    /// Get all values on local process
    virtual void get_local(std::vector<double>& values) const
    {
      dolfin_not_implemented();
    }

    /// Set all values on local process
    virtual void set_local(const std::vector<double>& values)
    {
      dolfin_not_implemented();
    }

    /// Add values to each entry on local process
    virtual void add_local(const Array<double>& values)
    {
      dolfin_not_implemented();
    }

    /// Gather entries into local vector x
    virtual void gather(GenericVector& x,
                        const std::vector<dolfin::la_index>& indices) const
    {
      dolfin_not_implemented();
    }

    /// Gather entries into x
    virtual void gather(std::vector<double>& x,
                        const std::vector<dolfin::la_index>& indices) const
    {
      dolfin_not_implemented();
    }

    /// Gather all entries into x on process 0
    virtual void gather_on_zero(std::vector<double>& x) const
    {
      dolfin_not_implemented();
    }

    /// Add multiple of given vector (AXPY operation)
    virtual void axpy(double a, const GenericVector& x)
    {
      dolfin_not_implemented();
    }

    /// Replace all entries in the vector by their absolute values
    virtual void abs()
    {
      dolfin_not_implemented();
    }

    /// Return inner product with given vector
    virtual double inner(const GenericVector& x) const
    {
      dolfin_not_implemented();
      return 0.0;
    }

    /// Return norm of vector
    virtual double norm(std::string norm_type) const
    {
      dolfin_not_implemented();
      return 0.0;
    }

    /// Return minimum value of vector
    virtual double min() const
    {
      dolfin_not_implemented();
      return 0.0;
    }

    /// Return maximum value of vector
    virtual double max() const
    {
      dolfin_not_implemented();
      return 0.0;
    }

    /// Return sum of vector
    virtual double sum() const
    {
      dolfin_not_implemented();
      return 0.0;
    }

    /// Return sum of selected rows in vector. Repeated entries are
    /// only summed once.
    virtual double sum(const Array<std::size_t>& rows) const
    {
      dolfin_not_implemented();
      return 0.0;
    }

    /// Multiply vector by given number
    virtual const VectorView& operator*= (double a)
    {
      dolfin_not_implemented();
      return *this;
    }

    /// Multiply vector by another vector pointwise
    virtual const VectorView& operator*= (const GenericVector& x)
    {
      dolfin_not_implemented();
      return *this;
    }

    /// Divide vector by given number
    virtual const VectorView& operator/= (double a)
    {
      dolfin_not_implemented();
      return *this;
    }

    /// Add given vector
    virtual const VectorView& operator+= (const GenericVector& x)
    {
      dolfin_not_implemented();
      return *this;
    }

    /// Add number to all components of a vector
    virtual const VectorView& operator+= (double a)
    {
      dolfin_not_implemented();
      return *this;
    }

    /// Subtract given vector
    virtual const VectorView& operator-= (const GenericVector& x)
    {
      dolfin_not_implemented();
      return *this;
    }

    /// Subtract number from all components of a vector
    virtual const VectorView& operator-= (double a)
    {
      dolfin_not_implemented();
      return *this;
    }

    /// Assignment operator
    // TODO: Shouldn't be disabled?!
    virtual const GenericVector& operator= (const GenericVector& x)
    {
      dolfin_not_implemented();
      *this = as_type<const VectorView>(x);
      return *this;
    }

    /// Assignment operator
    virtual const VectorView& operator= (double a)
    {
      dolfin_not_implemented();
    }

  private:

    /// Assignment operator
    virtual const VectorView& operator= (const VectorView& A) { }

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
    { dolfin_assert(_x);
      return _x->mpi_comm();}

    /// Return linear algebra backend factory
    virtual GenericLinearAlgebraFactory& factory() const
    { dolfin_not_implemented();
      return DefaultFactory::factory(); }

  private:

    std::shared_ptr<GenericVector> _x;

    std::size_t _dim;

    Array<la_index> _inds;

    std::vector<la_index> _work;

  };

}

#endif
