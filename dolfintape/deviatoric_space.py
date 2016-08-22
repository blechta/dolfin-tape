# Copyright (C) 2015-2016 Jan Blechta
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

import ufl
from dolfin import cpp


class TensorElement(ufl.TensorElement):
    """Monkey-patched version of ufl.TensorElement with additional kwarg
    zero_trace. If zero_trace==True, returns ZeroTraceTensorElement.
    """
    def __new__(cls, *args, **kwargs):
        if kwargs.pop("zero_trace", False):
            return ZeroTraceTensorElement(*args, **kwargs)
        else:
            return ufl.TensorElement(*args, **kwargs)


class ZeroTraceTensorElement(ufl.MixedElement):
    """This class just creates ufl.VectorElement of appropriate value
    dimension for holding dofs of matrix with zero trace. No mapping of
    vector dofs to matrix dofs is provided. User must introduce mapping
    manually into his forms using 'deviatoric' function.
    """
    def __new__(cls, *args, **kwargs):
        if len(args) <= 2 or not isinstance(args[1], ufl.Cell):
            cpp.dolfin_error("deviatoric_space.py",
                             "create tensor element with zero trace",
                             "Expected ufl.Cell as argument 2")
        if not kwargs.pop("symmetry", False):
            raise NotImplementedError, "Unsymmetric ZeroTraceTensorElement" \
                    " not implemented!"
        assert not kwargs.get("dim")
        kwargs["dim"] = cls._num_components(args[1].geometric_dimension())
        e = ufl.VectorElement(*args, **kwargs)
        e.__class__ = ZeroTraceTensorElement
        return e

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _num_components(gdim):
        return {2: 2, 3: 5}[gdim]


def deviatoric(vector):
    """Maps values of ZeroTraceTensorElement from flattened vector to
    symmetric matrix with zero trace.
    """
    if len(vector) == 2: # 2D
        return ufl.as_tensor(((vector[0],  vector[1]),
                              (vector[1], -vector[0])))
    if len(vector) == 5: # 3D
        return ufl.as_tensor(((vector[0], vector[2],  vector[3]),
                              (vector[2], vector[1],  vector[4]),
                              (vector[3], vector[4], -vector[0]-vector[1])))
    else:
        raise NotImplementedError
