from dolfin import compile_extension_module, Form, assemble
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
    def decorated_assemble(*args, **kwargs):
        tensor = kwargs.get('tensor')
        if isinstance(tensor, (VectorView, MatrixView)):
            [tensor.resize_work_array(dim,
                args[0].arguments()[dim].function_space().dofmap().max_cell_dimension())
                for dim in range(Form(args[0]).rank())]
        return assemble_function(*args, **kwargs)
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
