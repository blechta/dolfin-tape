from dolfin import *

from common.deviatoric_space import TensorFunctionSpace, deviatoric
from problems.GeneralizedStokes import GeneralizedStokesProblem

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
    u_ex = Expression(('+pow(sin(n*pi*x[0]), 2) * sin(2.0*n*pi*x[1])',
                       '-pow(sin(n*pi*x[1]), 2) * sin(2.0*n*pi*x[0])'),
                      n=n, degree=6)
    p_ex = Expression('0.0')
    s_ex = Expression(
            (('2.0*mu*n*pi*sin(2.0*n*pi*x[0])*sin(2.0*n*pi*x[1])',
              '2.0*mu*n*pi*( pow(sin(n*pi*x[0]), 2)*cos(2.0*n*pi*x[1])  '
              '     - pow(sin(n*pi*x[1]), 2)*cos(2.0*n*pi*x[0]) )'),
             ('2.0*mu*n*pi*( pow(sin(n*pi*x[0]), 2)*cos(2.0*n*pi*x[1])  '
              '     - pow(sin(n*pi*x[1]), 2)*cos(2.0*n*pi*x[0]) )',
              '-2.0*mu*n*pi*sin(2.0*n*pi*x[0])*sin(2.0*n*pi*x[1])')),
             mu=mu, n=n, degree=6)
    f = Expression(('+2.0*mu*n*n*pi*pi*( 2.0*pow(sin(n*pi*x[0]), 2) - cos(2.0*n*pi*x[0]) ) * sin(2.0*n*pi*x[1])',
                    '-2.0*mu*n*n*pi*pi*( 2.0*pow(sin(n*pi*x[1]), 2) - cos(2.0*n*pi*x[1]) ) * sin(2.0*n*pi*x[0])'),
                   mu=mu, n=n, degree=6)

    def __init__(self, N):
        mesh = UnitSquareMesh(N, N, "crossed")
        constitutive_law = NewtonianFluid(self.mu)
        GeneralizedStokesProblem.__init__(self, mesh, constitutive_law, self.f)

        # Attach domain and element to exact expressions
        # FIXME: This is a hack
        for e in [self.u_ex, self.p_ex, self.s_ex]:
            e._element = e._ufl_element = e.element().reconstruct(domain=mesh)
