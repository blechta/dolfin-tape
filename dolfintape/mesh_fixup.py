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

from dolfin import CellFunction, facets, Cell, vertices, refine

__all__ = ['mesh_fixup']


def mesh_fixup(mesh):
    """Refine cells which have all vertices on boundary and
    return a new mesh."""
    cf = CellFunction('bool', mesh)

    tdim = mesh.topology().dim()
    mesh.init(tdim-1, tdim)

    for f in facets(mesh):
        # Boundary facet?
        # TODO: Here we could check supplied facet function or subdomain
        if not f.exterior():
            continue

        # Pick adjacent cell
        c = Cell(mesh, f.entities(tdim)[0])

        # Number of vertices on boundary
        num_bad_vertices = sum(1 for v in vertices(c)
                               if any(fv.exterior() for fv in facets(v)))
        assert num_bad_vertices <= c.num_vertices()

        # Refine cell if all vertices are on boundary
        if num_bad_vertices == c.num_vertices():
            cf[c] = True

    return refine(mesh, cf)
