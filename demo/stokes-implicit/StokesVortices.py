from dolfin import *

from common.deviatoric_space import TensorFunctionSpace, deviatoric
from GeneralizedStokes import GeneralizedStokesProblem
from exact_solutions import pStokes_vortices

__all__ = ['StokesVortices']


class NewtonianFluid(object):
    def __init__(self, mu):
        self._mu = mu

    def r(self):
        return 2

    def mu(self):
        return self._mu

    def g(self):
        return lambda s, d: s - 2.0*Constant(self._mu)*d


class StokesVortices(GeneralizedStokesProblem):
    n = 4 # Number of vortices
    mu = 1.0

    def __init__(self, N):
        mesh = UnitSquareMesh(N, N, "crossed")
        constitutive_law = NewtonianFluid(self.mu)
        self.u_ex, self.p_ex, self.s_ex, self.f = \
            pStokes_vortices(n=self.n, mu=self.mu, r=2, eps=0.0,
                             degree=6, domain=mesh)
        GeneralizedStokesProblem.__init__(self, mesh, constitutive_law, self.f)
