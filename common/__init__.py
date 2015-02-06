# Avoid PETSc being initialized by DOLFIN, which sets some performance
# degrading parameters since e91b4100. This assumes that this module
# is imported before DOLFIN; otherwise the assertion may fail.
# TODO: Test whether it works!
from petsc4py import PETSc
from dolfin import SubSystemsManager
assert not SubSystemsManager.responsible_petsc()

# Enable info_{green,red,blue}
import ufl
ufl.set_level(ufl.INFO)


from flux_reconstructor import FluxReconstructor

__all__ = ['FluxReconstructor']
