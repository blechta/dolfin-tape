from dolfin import compile_extension_module, Form, assemble
import os

__all__ = ['MatrixView', 'VectorView', 'assemble']

"""This decorator ensures that *args are stored within class instance
at construction so that they are not garbage collected. This is not
ensured by C++ implementation."""
def store_args_decorator(cls):
    class Decorated(cls):
        def __init__(self, *args, **kwargs):
            self.__args = args
            super(Decorated, self).__init__(*args, **kwargs)
    return Decorated

path = os.path.realpath(
         os.path.join(os.getcwd(), os.path.dirname(__file__)))

code = open(path+"/MatrixView.h").read()
module = compile_extension_module(code, cppargs='-g -O2')
MatrixView = store_args_decorator(module.MatrixView)

code = open(path+"/VectorView.h").read()
module = compile_extension_module(code, cppargs='-g -O2')
VectorView = store_args_decorator(module.VectorView)


# TODO: Clean this up and document it!
# TODO: Decorate similarly other DOLFIN assemble functions!
def assemble_decorator(assemble_function):
    def decorated_assemble(*args, **kwargs):
        tensor = kwargs.get('tensor')
        if isinstance(tensor, (VectorView, MatrixView)):
            [tensor.resize_work_array(dim,
                args[0].arguments()[dim].function_space().dofmap().max_cell_dimension())
                for dim in range(Form(args[0]).rank())]
        return assemble_function(*args, **kwargs)
    return decorated_assemble

assemble = assemble_decorator(assemble)
