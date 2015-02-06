from dolfin import Expression, FiniteElement

__all__ = ['CellDiameters']


def CellDiameters(mesh):
    # TODO: Document
    element = FiniteElement('Discontinuous Lagrange', mesh, 0)
    e = Expression(cell_diameters_cpp_code, element=element)
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
