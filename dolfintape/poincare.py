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

from dolfin import Mesh, Cell, Vertex, cells, facets, edges, vertices, \
        warning, pi

from dolfintape.hat_function import hat_function_grad

__all__ = ['poincare_const', 'friedrichs_const', 'poincare_friedrichs_cutoff']


def poincare_const(o, p):
    if isinstance(o, Mesh):
        raise NotImplementedError("Poincare constant not implemented on mesh!")

    if isinstance(o, Cell):
        #assert isinstance(o, IntervalCell) or isinstance(o, TriangleCell) \
        #        or isinstance(o, TetrahedronCell), "Poincare constant not " \
        #        "implemented on non-simplicial cells!"
        d = max(e.length() for e in edges(o))
        return d*_poincare_simplex(p)

    if isinstance(o, type) and issubclass(o, Cell):
        #assert issubclass(o, IntervalCell) or issubclass(o, TriangleCell) \
        #        or issubclass(o, TetrahedronCell), "Poincare constant not " \
        #        "implemented on non-simplicial cells!"
        return _poincare_simplex(p)

    if isinstance(o, Vertex):
        # TODO: fix using ghosted mesh
        not_working_in_parallel("Poincare computation on patch")
        d = max(v0.point().distance(v1.point())
                for c0 in cells(o) for v0 in vertices(c0)
                for c1 in cells(o) for v1 in vertices(c1))
        # FIXME: Implement a check for simpliciality of the patch
        warning("Assuming simplicial patch for computation of Poincare const!")
        return d*_poincare_simplex(p)

    raise NotImplementedError


def friedrichs_const(o, p):
    if isinstance(o, Vertex):
        # TODO: fix using ghosted mesh
        not_working_in_parallel("Friedrichs computation on patch")
        d = max(v0.point().distance(v1.point())
                for c0 in cells(o) for v0 in vertices(c0)
                for c1 in cells(o) for v1 in vertices(c1))
        # FIXME: Implement the check
        warning("Friedrichs: assuming zero boundary of patch visible from any point!")
        return d

    raise NotImplementedError


def poincare_friedrichs_cutoff(o, p):
    if isinstance(o, Mesh):
        # TODO: easy fix - ghosted mesh + missing reduction
        not_working_in_parallel("PF cutoff on mesh")
        return max(poincare_friedrichs_cutoff(v, p) for v in vertices(o))

    if isinstance(o, Vertex):
        # TODO: fix using ghosted mesh
        not_working_in_parallel("PF cutoff on patch")
        hat_fun_grad = max(hat_function_grad(o, c) for c in cells(o))
        if any(f.exterior() for f in facets(o)):
            return 1.0 + friedrichs_const(o, p) * hat_fun_grad
        else:
            return 1.0 + poincare_const(o, p) * hat_fun_grad

    raise NotImplementedError


def _poincare_simplex(p):
    if p==1.0:
        # [Acosta, Duran 2003]
        return 0.5
    if p==2.0:
        # [Payne, Weinberger 1960], [Bebendorf 2003]
        return 1.0/pi
    # [Chua, Wheeden 2006]
    return 2.0*(0.5*p)**(1.0/p)
