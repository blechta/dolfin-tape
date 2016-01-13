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

"""This module provides cell diameter on simplicial mesh.
The module must be imported collectively on COMM_WORLD!
"""

from dolfin import Expression, FiniteElement, compile_expressions
from dolfin.functions.expression import create_compiled_expression_class

from dolfintape.poincare import _is_simplex

__all__ = ['CellDiameters']


def CellDiameters(mesh):
    """Returns piece-wise constant Expression defined as cell diameter,
    i.e. maximum edge length for simplices."""
    assert _is_simplex(mesh.type())
    element = FiniteElement('Discontinuous Lagrange', mesh.ufl_cell(), 0)
    e = _cell_diameters_base_class(None, element=element, domain=mesh)
    return e


_cell_diameters_cpp_code="""
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

def _create_cell_diameters_base_class():
    """This functions builds PyDOLFIN compiled expression representing
    the cell diameters. An instance can be created as usual.

    The purpose of this procedure, contrary to usual

        Expression(cppcode, **kwargs)

    is to avoid JIT chain on dynamic class creation which may be too
    expensive in hot loops.

    NOTE: This function is collective on COMM_WORLD
          (trough compile_expressions).
    """
    cpp_base, members = compile_expressions([_cell_diameters_cpp_code])
    cpp_base, members = cpp_base[0], members[0]
    assert len(members) == 0
    base = create_compiled_expression_class(cpp_base)
    return base

_cell_diameters_base_class = _create_cell_diameters_base_class()
del _create_cell_diameters_base_class
