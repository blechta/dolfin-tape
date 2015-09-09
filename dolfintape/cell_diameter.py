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

from dolfin import Expression, FiniteElement

__all__ = ['CellDiameters']


def CellDiameters(mesh):
    """Returns piece-wise constant Expression defined as cell diameter,
    i.e. maximum edge length for simplices."""
    element = FiniteElement('Discontinuous Lagrange', mesh.ufl_cell(), 0)
    e = Expression(cell_diameters_cpp_code, element=element, domain=mesh)
    return e


cell_diameters_cpp_code="""
#include <dolfin/mesh/Edge.h>

namespace dolfin {

  class CellDiameters: public Expression
  {
  public:

    CellDiameters() : Expression() { }

    void restrict(double* w, const FiniteElement& element,
                  const Cell& cell,
                  const double* vertex_coordinates,
                  const ufc::cell& ufc_cell) const
    {
      *w = 0.0;
      for (EdgeIterator edge(cell); !edge.end(); ++edge)
        *w = std::max(edge->length(), *w);
    }
  };

}
"""
