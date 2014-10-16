from dolfin import as_backend_type, PETScMatrix, MPI, mpi_comm_world, tic, toc
import numpy as np

__all__ = ['la_index', 'num_nonzeros']

# TODO: What is the proper type for 64-bit PetscInt and how to determine it?
# NOTE: Seems that la_index is not mapped properly in DOLFIN SWIG wrappers,
#       see https://bitbucket.org/fenics-project/dolfin/issue/366
la_index = np.intc


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
