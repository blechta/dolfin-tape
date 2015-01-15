from dolfin import as_backend_type, PETScMatrix, MPI, mpi_comm_world, \
                   tic, toc, la_index_dtype
from mpi4py import MPI as MPI4py
import numpy as np

__all__ = ['la_index_mpitype', 'num_nonzeros']

def la_index_mpitype():
    try:
        mpi_typedict = MPI4py._typedict
    except AttributeError:
        mpi_typedict = MPI4py.__TypeDict__
    return mpi_typedict[np.dtype(la_index_dtype()).char]

def num_nonzeros(A):
    # TODO: Rewrite to C++ for performance
    # TODO: Is it efficient, querying the rows?
    A = as_backend_type(A)
    assert isinstance(A, PETScMatrix), "Provide PETScMatrix!"
    mat = A.mat()
    # TODO: fix in parallel
    if MPI.size(mpi_comm_world()) > 1:
        return
    num_rows = mat.getSize()[0]
    tic()
    num_nonzeros = sum(len(mat.getRow(i)[0]) for i in xrange(num_rows))
    return num_nonzeros, toc()
