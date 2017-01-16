# Copyright (C) 2015 Jan Blechta
#
# This file is part of dolfin-tape.
#
# dolfin-tape is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# dolfin-tape is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with dolfin-tape. If not, see <http://www.gnu.org/licenses/>.

"""This is dolfin-tape, the DOLFIN tools for a posteriori error estimation."""

__author__ = "Jan Blechta"
__version__ = "2016.2.0"
__license__ = 'LGPL v3'

# Avoid PETSc being initialized by DOLFIN, which sets some performance
# degrading parameters since e91b4100. This assumes that this module
# is imported before DOLFIN; otherwise the assertion may fail.
# TODO: Test whether it works!
from petsc4py import PETSc
from dolfin import SubSystemsManager
assert not SubSystemsManager.responsible_petsc()
del PETSc, SubSystemsManager

# Reduce DOLFIN logging bloat in parallel
from dolfin import set_log_level, get_log_level, MPI, mpi_comm_world
set_log_level(get_log_level()+(0 if MPI.rank(mpi_comm_world())==0 else 1))
del set_log_level, get_log_level

# Parse command-line options
# FIXME: Automatic parsing temporarily commented out as it disallows parsing
#        our own application specific parameters. Do we need it somewhere?
#from dolfin import parameters
#parameters.parse()

# Enable info_{green,red,blue} on rank 0
import ufl
ufl.set_level(ufl.INFO if MPI.rank(mpi_comm_world())==0 else ufl.INFO+1)
del ufl, MPI, mpi_comm_world


from dolfintape.flux_reconstructor import FluxReconstructor
from dolfintape.cell_diameter import CellDiameters
from dolfintape.mesh_diameter import mesh_diameter

__all__ = ['FluxReconstructor', 'CellDiameters', 'mesh_diameter']
