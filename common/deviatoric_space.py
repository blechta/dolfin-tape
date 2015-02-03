import dolfin
from dolfin import cpp

# TODO: This was intended as thin wrapper above DOLFIN providing function
#       spaces of traceless tensors and transparent manipulation with them.
#       Following should work for single traceless tensor space but will
#       not work with mixed spaces were only subspace needs mapping between
#       independent components and traceless tensor components.


def TensorFunctionSpace(*args, **kwargs):
    if kwargs.pop("zero_trace", False):
        return ZeroTraceTensorFunctionSpace(*args, **kwargs)
    else:
        return dolfin.TensorFunctionSpace(*args, **kwargs)


class ZeroTraceTensorFunctionSpace(dolfin.VectorFunctionSpace):
    def __init__(self, mesh, *args, **kwargs):
        if not isinstance(mesh, (cpp.Mesh, cpp.Restriction)):
            cpp.dolfin_error("functionspace.py",
                             "create function space",
                             "Illegal argument, not a mesh or restriction: " + str(mesh))
        if not kwargs.pop("symmetry", False):
            raise NotImplementedError, "Unsymmetric ZeroTraceTensorFunctionSpace" \
                    " not implemented!"
        kwargs["dim"] = self._num_components(mesh.geometry().dim())
        dolfin.VectorFunctionSpace.__init__(self, mesh, *args, **kwargs)

    def _num_components(self, gdim):
        return {2: 2, 3: 5}[gdim]


def deviatoric(vector):
    if len(vector) == 2: # 2D
        return dolfin.as_tensor(((vector[0],  vector[1]),
                                 (vector[1], -vector[0])))
    if len(vector) == 5: # 3D
        return dolfin.as_tensor(((vector[0], vector[2],  vector[3]),
                                 (vector[2], vector[1],  vector[4]),
                                 (vector[3], vector[4], -vector[0]-vector[1])))
    else:
        raise NotImplementedError


def TrialFunctions(V):
    if isinstance(V, ZeroTraceTensorFunctionSpace):
        return deviatoric(dolfin.TrialFunctions(V))
    else:
        return dolfin.TrialFunctions(V)


def TestFunctions(V):
    if isinstance(V, ZeroTraceTensorFunctionSpace):
        return deviatoric(dolfin.TestFunctions(V))
    else:
        return dolfin.TestFunctions(V)
