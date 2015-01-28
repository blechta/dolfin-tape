from dolfin import la_index_dtype
from mpi4py import MPI as MPI4py
import numpy as np

__all__ = ['la_index_mpitype']

def la_index_mpitype():
    """mpi4py type corresponding to dolfin::la_index."""
    try:
        mpi_typedict = MPI4py._typedict
    except AttributeError:
        mpi_typedict = MPI4py.__TypeDict__
    return mpi_typedict[np.dtype(la_index_dtype()).char]
