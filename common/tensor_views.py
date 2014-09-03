from dolfin import compile_extension_module
import os

__all__ = ['MatrixView', 'VectorView']

"""This decorator ensures that *args are stored within class instance
at construction so that they are not garbage collected. This is not
ensured by C++ implementationi."""
def store_args_decorator(cls):
    class Decorated(cls):
        def __init__(self, *args, **kwargs):
            self.__args = args
            super(Decorated, self).__init__(*args, **kwargs)
    return Decorated

path = os.path.realpath(
         os.path.join(os.getcwd(), os.path.dirname(__file__)))

code = open(path+"/MatrixView.h").read()
module = compile_extension_module(code, cppargs='-g3')
MatrixView = store_args_decorator(module.MatrixView)

code = open(path+"/VectorView.h").read()
module = compile_extension_module(code, cppargs='-g3')
VectorView = store_args_decorator(module.VectorView)
