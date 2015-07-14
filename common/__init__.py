"""This is dolfin-tape, the DOLFIN tools for a posteriori error estimation."""

__author__ = "Jan Blechta"
__version__ = "1.6.0dev"
__license__ = 'LGPL v3'

# Avoid PETSc being initialized by DOLFIN, which sets some performance
# degrading parameters since e91b4100. This assumes that this module
# is imported before DOLFIN; otherwise the assertion may fail.
# TODO: Test whether it works!
from petsc4py import PETSc
from dolfin import SubSystemsManager
assert not SubSystemsManager.responsible_petsc()

# Reduce DOLFIN logging bloat in parallel
from dolfin import set_log_level, get_log_level, MPI, mpi_comm_world
set_log_level(get_log_level()+(0 if MPI.rank(mpi_comm_world())==0 else 1))

# Parse command-line options
from dolfin import parameters
parameters.parse()

# Enable info_{green,red,blue} on rank 0
import ufl
ufl.set_level(ufl.INFO if MPI.rank(mpi_comm_world())==0 else ufl.INFO+1)


from flux_reconstructor import FluxReconstructor

__all__ = ['FluxReconstructor']
