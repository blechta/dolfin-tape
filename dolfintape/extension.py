# Copyright (C) 2016 Jan Blechta
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

"""This module provides zero extension of function.
The module must be imported collectively on COMM_WORLD!
"""

from dolfin import Expression, FiniteElement, compile_expressions
from dolfin.functions.expression import create_compiled_expression_class

__all__ = ['Extension']


def Extension(u, domain):
    """Returns Expression representing zero extension of u to given domain"""
    element = u.ufl_element()
    mesh = u.function_space().mesh()
    e = _extension_base_class(None, element=element, domain=domain)
    e.u = u
    return e


_extension_cpp_code = """
#include <dolfin/common/utils.h>

namespace dolfin {

  class Extension : public Expression
  {
  public:

    std::shared_ptr<Function> u;

    Extension() : Expression() { }

    void eval(Array<double>& values, const Array<double>& x) const
    {
      dolfin_assert(u);
      try
      {
        u->eval(values, x);
      }
      catch (std::runtime_error &e)
      {
        zerofill(values.data(), values.size());
        return;
      }
      dolfin_assert(!u->get_allow_extrapolation());
    }
  };

}
"""

def _create_extension_base_class():
    """This functions builds PyDOLFIN compiled expression representing
    the zero extension of function. An instance can be created as usual.

    The purpose of this procedure, contrary to usual

        Expression(cppcode, **kwargs)

    is to avoid JIT chain on dynamic class creation which may be too
    expensive in hot loops.

    NOTE: This function is collective on COMM_WORLD
          (trough compile_expressions).
    """
    cpp_base, members = compile_expressions([_extension_cpp_code])
    cpp_base, members = cpp_base[0], members[0]
    assert len(members) == 0
    base = create_compiled_expression_class(cpp_base)
    return base

_extension_base_class = _create_extension_base_class()
del _create_extension_base_class
