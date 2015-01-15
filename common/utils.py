from dolfin import as_backend_type, PETScMatrix, MPI, mpi_comm_world, \
                   tic, toc, la_index_dtype
from mpi4py import MPI as MPI4py
import numpy as np

__all__ = ['la_index_mpitype']

def la_index_mpitype():
    try:
        mpi_typedict = MPI4py._typedict
    except AttributeError:
        mpi_typedict = MPI4py.__TypeDict__
    return mpi_typedict[np.dtype(la_index_dtype()).char]
