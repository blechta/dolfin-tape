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

import dolfin
from dolfin import cpp

# TODO: This was intended as thin wrapper above DOLFIN providing function
#       spaces of traceless tensors and transparent manipulation with them.
#       Following should work for single traceless tensor space but will
#       not work with mixed spaces were only subspace needs mapping between
#       independent components and traceless tensor components.


def TensorFunctionSpace(*args, **kwargs):
    if kwargs.pop("zero_trace", False):
        return ZeroTraceTensorFunctionSpace(*args, **kwargs)
    else:
        return dolfin.TensorFunctionSpace(*args, **kwargs)


class ZeroTraceTensorFunctionSpace(dolfin.FunctionSpace):
    def __new__(cls, mesh, *args, **kwargs):
        if not isinstance(mesh, cpp.Mesh):
            cpp.dolfin_error("functionspace.py",
                             "create function space",
                             "Illegal argument, not a mesh: " + str(mesh))
        if not kwargs.pop("symmetry", False):
            raise NotImplementedError, "Unsymmetric ZeroTraceTensorFunctionSpace" \
                    " not implemented!"
        assert not kwargs.get("dim")
        kwargs["dim"] = cls._num_components(mesh.geometry().dim())
        V = dolfin.VectorFunctionSpace(mesh, *args, **kwargs)
        V.__class__ = ZeroTraceTensorFunctionSpace
        return V

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _num_components(gdim):
        return {2: 2, 3: 5}[gdim]


def deviatoric(vector):
    if len(vector) == 2: # 2D
        return dolfin.as_tensor(((vector[0],  vector[1]),
                                 (vector[1], -vector[0])))
    if len(vector) == 5: # 3D
        return dolfin.as_tensor(((vector[0], vector[2],  vector[3]),
                                 (vector[2], vector[1],  vector[4]),
                                 (vector[3], vector[4], -vector[0]-vector[1])))
    else:
        raise NotImplementedError


def TrialFunctions(V):
    if isinstance(V, ZeroTraceTensorFunctionSpace):
        return deviatoric(dolfin.TrialFunctions(V))
    else:
        return dolfin.TrialFunctions(V)


def TestFunctions(V):
    if isinstance(V, ZeroTraceTensorFunctionSpace):
        return deviatoric(dolfin.TestFunctions(V))
    else:
        return dolfin.TestFunctions(V)
