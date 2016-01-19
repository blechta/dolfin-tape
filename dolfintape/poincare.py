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

from functools import wraps

from dolfin import Mesh, Cell, Vertex, cells, facets, edges, vertices, \
        pi, not_working_in_parallel, CellType, Event, get_log_level, set_log_level, INFO

from dolfintape.hat_function import hat_function_grad


__all__ = ['poincare_const', 'friedrichs_const', 'poincare_friedrichs_cutoff']


def poincare_const(o, p, d=1):
    # Vectorial Poincare, see [Blechta, Malek, Vohralik 2016]
    if d != 1 and p != 2.0:
        return d**abs(0.5-1.0/p) * poincare_const(o, p)

    if isinstance(o, Mesh):
        raise NotImplementedError("Poincare constant not implemented on mesh!")

    if isinstance(o, Cell):
        assert _is_simplex(o), "Poincare constant not " \
                "implemented on non-simplicial cells!"
        h = max(e.length() for e in edges(o))
        return h*_poincare_convex(p)

    if isinstance(o, CellType):
        assert _is_simplex(o), "Poincare constant not " \
                "implemented on non-simplicial cells!"
        return _poincare_convex(p)

    if isinstance(o, Vertex):
        # TODO: fix using ghosted mesh
        not_working_in_parallel("Poincare computation on patch")
        h = max(v0.point().distance(v1.point())
                for c0 in cells(o) for v0 in vertices(c0)
                for c1 in cells(o) for v1 in vertices(c1))
        # FIXME: Implement a check for convexity of the patch
        _warn_poincare_convex()
        return h*_poincare_convex(p)

    raise NotImplementedError


def friedrichs_const(o, p):
    if isinstance(o, Vertex):
        # TODO: fix using ghosted mesh
        not_working_in_parallel("Friedrichs computation on patch")
        d = max(v0.point().distance(v1.point())
                for c0 in cells(o) for v0 in vertices(c0)
                for c1 in cells(o) for v1 in vertices(c1))
        # FIXME: Implement the check
        _warn_friedrichs_lines()
        return d

    raise NotImplementedError


def _change_log_level(log_level, function):
    """This decorator wraps function with temporary change to given log level.
    """
    @wraps(function)
    def wrapped(*args, **kwargs):
        old_level = get_log_level()
        set_log_level(log_level)
        function(*args, **kwargs)
        set_log_level(old_level)
    return wrapped

# Event works only when log level <= INFO; we need more
Event.__call__ = _change_log_level(INFO, Event.__call__)

# Warnings with limited number of issues
_warn_poincare_convex = Event(
        "WARNING: Assuming convex patch for computation of Poincare const!")
_warn_friedrichs_lines = Event(
        "WARNING: Assuming zero boundary of patch visible from any point "
        "for computation of Friedrichs const!")


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


def _poincare_convex(p):
    if p==1.0:
        # [Acosta, Duran 2003]
        return 0.5
    if p==2.0:
        # [Payne, Weinberger 1960], [Bebendorf 2003]
        return 1.0/pi
    # [Chua, Wheeden 2006]
    return 2.0*(0.5*p)**(1.0/p)


_simplices = [CellType.interval, CellType.triangle, CellType.tetrahedron]
def _is_simplex(c):
    if isinstance(c, Cell):
        return c.type() in _simplices
    if isinstance(c, CellType):
        return c.cell_type() in _simplices
    raise NotImplementedError
