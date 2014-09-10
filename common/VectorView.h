#ifndef __VECTOR_VIEW_H
#define __VECTOR_VIEW_H

//#include <algorithm>
//#include <utility>
#include <vector>
#include <dolfin/log/log.h>
//#include <dolfin/common/types.h>
#include <dolfin/common/Array.h>
#include <dolfin/la/TensorLayout.h>
#include <dolfin/la/GenericTensor.h>

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
               const Array<std::size_t>& inds)
      : _x(x), _dim(dim),
        _inds(inds.size(), const_cast<std::size_t*>(inds.data()))
    {
      // TODO: check indices?
    }

    /// Copy constructor
    VectorView(const VectorView& vv)
      : _x(vv._x), _dim(vv._dim),
        _inds(vv._inds.size(), const_cast<std::size_t*>(vv._inds.data()))
    { }

    /// Destructor
    virtual ~VectorView() {}

    //--- Implementation of the GenericTensor interface ---

    /// Set all entries to zero and keep any sparse structure
    virtual void zero()
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Finalize assembly of tensor
    virtual void apply(std::string mode)
    { dolfin_assert(_x);
      _x->apply(mode); }

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const
    {
      //TODO: implement!
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
    virtual void init(MPI_Comm comm, std::size_t N)
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Intitialize vector with given local ownership range
    virtual void init(MPI_Comm comm,
                      std::pair<std::size_t, std::size_t> range)
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Initialise vector with given ownership range and with ghost
    /// values
    virtual void init(MPI_Comm comm,
                      std::pair<std::size_t, std::size_t> range,
                      const std::vector<std::size_t>& local_to_global_map,
                      const std::vector<la_index>& ghost_indices)
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Return global size of vector
    virtual std::size_t size() const
    { return _dim; }

    /// Return local size of vector
    virtual std::size_t local_size() const
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Return local ownership range of a vector
    virtual std::pair<std::size_t, std::size_t> local_range() const
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Determine whether global vector index is owned by this process
    virtual bool owns_index(std::size_t i) const
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Get block of values using global indices (values must all live
    /// on the local process, ghosts cannot be accessed)
    virtual void get(double* block, std::size_t m,
                     const dolfin::la_index* rows) const
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Get block of values using local indices (values must all live
    /// on the local process, ghost are accessible)
    virtual void get_local(double* block, std::size_t m,
                           const dolfin::la_index* rows) const
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Set block of values using global indices
    virtual void set(const double* block, std::size_t m,
                     const dolfin::la_index* rows)
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Set block of values using local indices
    virtual void set_local(const double* block, std::size_t m,
                           const dolfin::la_index* rows)
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Add block of values using global indices
    virtual void add(const double* block, std::size_t m,
                     const dolfin::la_index* rows)
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Add block of values using local indices
    virtual void add_local(const double* block, std::size_t m,
                           const dolfin::la_index* rows)
    {
      // TODO: Dynamic allocation of memory?! Not good!
      std::vector<dolfin::la_index> inds;
      inds.resize(m);
      for (std::size_t i = 0; i < m; ++i)
        inds[i] = _inds[rows[i]];
      _x->add(block, m, inds.data());
    }

    /// Get all values on local process
    virtual void get_local(std::vector<double>& values) const
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Set all values on local process
    virtual void set_local(const std::vector<double>& values)
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Add values to each entry on local process
    virtual void add_local(const Array<double>& values)
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Gather entries into local vector x
    virtual void gather(GenericVector& x,
                        const std::vector<dolfin::la_index>& indices) const
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Gather entries into x
    virtual void gather(std::vector<double>& x,
                        const std::vector<dolfin::la_index>& indices) const
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Gather all entries into x on process 0
    virtual void gather_on_zero(std::vector<double>& x) const
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Add multiple of given vector (AXPY operation)
    virtual void axpy(double a, const GenericVector& x)
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Replace all entries in the vector by their absolute values
    virtual void abs()
    {
      //TODO: implement!
      dolfin_not_implemented();
    }

    /// Return inner product with given vector
    virtual double inner(const GenericVector& x) const
    {
      //TODO: implement!
      dolfin_not_implemented();
      return 0.0;
    }

    /// Return norm of vector
    virtual double norm(std::string norm_type) const
    {
      //TODO: implement!
      dolfin_not_implemented();
      return 0.0;
    }

    /// Return minimum value of vector
    virtual double min() const
    {
      //TODO: implement!
      dolfin_not_implemented();
      return 0.0;
    }

    /// Return maximum value of vector
    virtual double max() const
    {
      //TODO: implement!
      dolfin_not_implemented();
      return 0.0;
    }

    /// Return sum of vector
    virtual double sum() const
    {
      //TODO: implement!
      dolfin_not_implemented();
      return 0.0;
    }

    /// Return sum of selected rows in vector. Repeated entries are
    /// only summed once.
    virtual double sum(const Array<std::size_t>& rows) const
    {
      //TODO: implement!
      dolfin_not_implemented();
      return 0.0;
    }

    /// Multiply vector by given number
    virtual const VectorView& operator*= (double a)
    {
      //TODO: implement!
      dolfin_not_implemented();
      return *this;
    }

    /// Multiply vector by another vector pointwise
    virtual const VectorView& operator*= (const GenericVector& x)
    {
      //TODO: implement!
      dolfin_not_implemented();
      return *this;
    }

    /// Divide vector by given number
    virtual const VectorView& operator/= (double a)
    {
      //TODO: implement!
      dolfin_not_implemented();
      return *this;
    }

    /// Add given vector
    virtual const VectorView& operator+= (const GenericVector& x)
    {
      //TODO: implement!
      dolfin_not_implemented();
      return *this;
    }

    /// Add number to all components of a vector
    virtual const VectorView& operator+= (double a)
    {
      //TODO: implement!
      dolfin_not_implemented();
      return *this;
    }

    /// Subtract given vector
    virtual const VectorView& operator-= (const GenericVector& x)
    {
      //TODO: implement!
      dolfin_not_implemented();
      return *this;
    }

    /// Subtract number from all components of a vector
    virtual const VectorView& operator-= (double a)
    {
      //TODO: implement!
      dolfin_not_implemented();
      return *this;
    }

    /// Assignment operator
    // TODO: Shouldn't be disabled?!
    virtual const GenericVector& operator= (const GenericVector& x)
    {
      //TODO: implement!
      dolfin_not_implemented();
      *this = as_type<const VectorView>(x);
      return *this;
    }

    /// Assignment operator
    virtual const VectorView& operator= (double a)
    {
      //TODO: implement!
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
      //TODO: implement!
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

    Array<std::size_t> _inds;

  };

}

#endif
