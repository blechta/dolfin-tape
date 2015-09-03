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

from dolfin import Expression, cpp, FiniteElement, jit, \
        vertices, facets, Vertex, not_working_in_parallel

__all__ = ['hat_function_collection', 'hat_function', 'hat_function_grad']


def hat_function_collection(vertex_colors, color, element=None):
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

    e = Expression(hats_cpp_code, element=element)
    e.vertex_colors = vertex_colors
    e.color = color
    e.dolfin_element = dolfin_element

    return e


hats_cpp_code="""
class HatFunctionCollection : public Expression
{
public:

  std::shared_ptr<VertexFunction<std::size_t> > vertex_colors;

  std::size_t color;

  std::shared_ptr<FiniteElement> dolfin_element;

  HatFunctionCollection() : Expression() { }

  void restrict(double* w, const FiniteElement& element,
                const Cell& cell,
                const double* vertex_coordinates,
                const ufc::cell& ufc_cell) const
  {
    if ( cell.mesh_id() == vertex_colors->mesh()->id()
         && element.hash() == dolfin_element->hash() )
      for (VertexIterator v(cell); !v.end(); ++v)
        *(w++) = (*vertex_colors)[v->index()] == color ? 1.0 : 0.0;
    else
      restrict_as_ufc_function(w, element, cell, vertex_coordinates, ufc_cell);
  }
};
"""


def hat_function(vertex):
    """Return Expression on Lagrange degree 1 element which is
      1 ... at given 'vertex'
      0 ... at other vertices
    """
    assert isinstance(vertex, Vertex)
    mesh = vertex.mesh()
    element = FiniteElement('Lagrange', mesh, 1)
    ufc_element, ufc_dofmap = jit(element, mpi_comm=mesh.mpi_comm())
    dolfin_element = cpp.FiniteElement(ufc_element)

    e = Expression(hat_cpp_code, element=element)
    e.vertex = vertex
    e.dolfin_element = dolfin_element

    return e

hat_cpp_code="""
#include <dolfin/mesh/Vertex.h>

namespace dolfin {

class HatFunction : public Expression
{
public:

  MeshEntity vertex;

  std::shared_ptr<FiniteElement> dolfin_element;

  HatFunction() : Expression() { }

  void restrict(double* w, const FiniteElement& element,
                const Cell& cell,
                const double* vertex_coordinates,
                const ufc::cell& ufc_cell) const
  {
    if ( cell.mesh_id() == vertex.mesh_id()
         && element.hash() == dolfin_element->hash() )
      for (VertexIterator v(cell); !v.end(); ++v)
        *(w++) = *v == vertex ? 1.0 : 0.0;
    else
      restrict_as_ufc_function(w, element, cell, vertex_coordinates, ufc_cell);
  }
};
}
"""


def hat_function_grad(vertex, cell):
    """Compute L^\infty-norm of gradient of hat function on 'cell'
    and value 1 in 'vertex'."""
    # TODO: fix using ghosted mesh
    not_working_in_parallel("function 'hat_function_grad'")

    assert vertex in vertices(cell), "vertex not in cell!"

    # Find adjacent facet
    f = [f for f in facets(cell) if not vertex in vertices(f)]
    assert len(f) == 1, "Something strange with adjacent cell!"
    f = f[0]

    # Get unit normal
    n = f.normal()
    n /= n.norm()

    # Pick some vertex on facet
    # FIXME: Is it correct index in parallel?
    facet_vertex_0 = Vertex(cell.mesh(), f.entities(0)[0])

    # Compute signed distance from vertex to facet plane
    d = (facet_vertex_0.point() - vertex.point()).dot(n)

    # Return norm of gradient
    assert d != 0.0, "Degenerate cell!"
    return 1.0/abs(d)
