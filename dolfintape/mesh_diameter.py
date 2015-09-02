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

from dolfin import not_working_in_parallel, facets, vertices

__all__ = ['mesh_diameter']


def mesh_diameter(mesh):
    # FIXME: Quadratic algorithm is too slow!
    """Return mesh diameter, i.e. \sup_{x,y \in mesh} |x-y|. Algorithm
    loops quadratically over boundary facets."""

    not_working_in_parallel("Function 'mesh_diameter'")
    assert mesh.topology().dim() == mesh.geometry().dim(), \
            "Function 'mesh_diameter' not working on manifolds."

    tdim = mesh.topology().dim()
    mesh.init(tdim-1, tdim)

    diameter = 0.0

    for f0 in facets(mesh):
        if not f0.exterior():
            continue

        for f1 in facets(mesh):
            if not f1.exterior():
                continue

            for v0 in vertices(f0):
                for v1 in vertices(f1):
                    diameter = max(diameter, v0.point().distance(v1.point()))

    return diameter
