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

from dolfin import Expression, cpp, FiniteElement, jit

__all__ = ['hat_function']


def hat_function(vertex_colors, color, element=None):
    """Return Expression on given element which takes values:
      1 ... if vertex_colors[node] == color
      0 ... at other nodes
    This is well defined just on Lagrange 1 element (default) and Dicontinuous
    Lagrange 1 element.

    NOTE: This expression provides a little hack as it lacks continuity across
    MPI partitions boundaries unless vertex_colors is compatible there. In fact,
    this behaviour is needed in FluxReconstructor."""
    assert isinstance(vertex_colors, cpp.VertexFunctionSizet)
    mesh = vertex_colors.mesh()
    if not element:
        element = FiniteElement('Lagrange', mesh, 1)
    assert element.family() in ['Lagrange', 'Discontinuous Lagrange']
    assert element.degree() == 1
    ufc_element, ufc_dofmap = jit(element, mpi_comm=mesh.mpi_comm())
    dolfin_element = cpp.FiniteElement(ufc_element)

    e = Expression(hat_cpp_code, element=element)
    e.vertex_colors = vertex_colors
    e.color = color
    e.dolfin_element = dolfin_element

    return e


hat_cpp_code="""
class HatFunction : public Expression
{
public:

  std::shared_ptr<VertexFunction<std::size_t> > vertex_colors;

  std::size_t color;

  std::shared_ptr<FiniteElement> dolfin_element;

  HatFunction() : Expression() { }

  void restrict(double* w, const FiniteElement& element,
                const Cell& cell,
                const double* vertex_coordinates,
                const ufc::cell& ufc_cell) const
  {
    dolfin_assert(&cell.mesh() == &(*vertex_colors->mesh()));
    dolfin_assert(element.hash() == dolfin_element->hash());

    for (VertexIterator vertex(cell); !vertex.end(); ++vertex)
      *(w++) = (*vertex_colors)[vertex->index()] == color ? 1.0 : 0.0;
  }
};
"""
