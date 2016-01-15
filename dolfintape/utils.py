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

from dolfin import la_index_dtype, compile_extension_module
from mpi4py import MPI as MPI4py
import numpy as np
import os, errno

__all__ = ['la_index_mpitype', 'adapt', 'PETScVector_ipow', 'pow']

def la_index_mpitype():
    """mpi4py type corresponding to dolfin::la_index."""
    try:
        mpi_typedict = MPI4py._typedict
    except AttributeError:
        mpi_typedict = MPI4py.__TypeDict__
    return mpi_typedict[np.dtype(la_index_dtype()).char]


adapt_wrapper_code = """
#include <dolfin/adaptivity/adapt.h>

namespace dolfin {

std::shared_ptr<MeshFunction<std::size_t>> adapt_wrapper(
  const MeshFunction<std::size_t>& mesh_function,
  std::shared_ptr<const Mesh> adapted_mesh)
{
  std::shared_ptr<MeshFunction<std::size_t>> mf;
  mf = std::make_shared<MeshFunction<std::size_t>>(adapt(mesh_function, adapted_mesh));
  return mf;
}
}
"""

adapt = compile_extension_module(adapt_wrapper_code).adapt_wrapper


pow_wrapper_code = """
#include <petscvec.h>
#include <dolfin/la/PETScVector.h>

namespace dolfin {

void PETScVector_pow(PETScVector& x, double p)
{
  dolfin_assert(x.vec());
  PetscErrorCode ierr = VecPow(x.vec(), p);
  if (ierr != 0) x.petsc_error(ierr, "utils.py", "VecPow");
}
}
"""

# Power inplace
PETScVector_ipow = compile_extension_module(pow_wrapper_code).PETScVector_pow

import __builtin__
from dolfin import PETScVector
# Power
def pow(*args):
    try:
        return __builtin__.pow(*args)
    except TypeError:
        assert len(args) == 2 and isinstance(args[0], PETScVector)
        x = PETScVector(args[0])
        PETScVector_ipow(x, args[1])
        return x
pow.__doc__ = __builtin__.pow.__doc__ + "\n\n" + PETScVector_ipow.__doc__


def mkdir_p(name):
    """$ mkdir -p name"""
    try:
        os.mkdir(name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e
