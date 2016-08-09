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

from dolfin import compile_extension_module, assemble
from dolfin.fem.assembling import _create_dolfin_form

from functools import wraps
import os

__all__ = ['MatrixView', 'VectorView', 'assemble']


def _store_args_decorator(cls):
    """This decorator ensures that *args to __init__ are stored within class
    instance at construction so that they are not garbage collected. This is
    helpful for controlling lifetime of constructor arguments when not handled
    on C++ side, e.g. for numpy.ndarray --> dolfin::Array typemap."""
    class Decorated(cls):
        def __init__(self, *args, **kwargs):
            self.__args = args
            super(Decorated, self).__init__(*args, **kwargs)
    Decorated.__name__ = "%s_storedArgs" % cls.__name__
    return Decorated


# TODO: Decorate similarly other DOLFIN assemble functions!
def _assemble_decorator(assemble_function):
    """This decorator wraps DOLFIN assemble function for automatic use with
    Matrix/VectorView. It merely checks maximal cell dimension of underlying
    dofmaps in supplied forms and resizes work arrays of tensor views
    appropriately."""
    @wraps(assemble_function)
    def decorated_assemble(*args, **kwargs):
        tensor = kwargs.get('tensor')
        if isinstance(tensor, (VectorView, MatrixView)):
            form = _create_dolfin_form(args[0], kwargs.pop("form_compiler_parameters", None))
            spaces = form.function_spaces
            for i in range(len(spaces)):
                dim = spaces[i].dofmap().max_cell_dimension()
                tensor.resize_work_array(i, dim)
        else:
            form = args[0]
        return assemble_function(form, *args[1:], **kwargs)
    return decorated_assemble


# Find absolute path of this module
path = os.path.realpath(
         os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Load MatrixView
code = open(path+"/MatrixView.h").read()
module = compile_extension_module(code, cppargs='-g -O2')
MatrixView = _store_args_decorator(module.MatrixView)

# Load VectorView
code = open(path+"/VectorView.h").read()
module = compile_extension_module(code, cppargs='-g -O2')
VectorView = _store_args_decorator(module.VectorView)

# Provide assemble function
assemble = _assemble_decorator(assemble)
