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

from dolfin import *

from dolfintape.demo_problems.GeneralizedStokes import GeneralizedStokesProblem
from dolfintape.demo_problems.exact_solutions import pStokes_vortices

__all__ = ['PowerLawVortices']


class PowerLawFluid(object):
    def __init__(self, mu, r):
        self._mu = mu
        self._r = r

    def r(self):
        return self._r

    def mu(self):
        return self._mu

    def g(self):
        r, mu = self._r, self._mu
        return lambda s, d, eps: (
          Constant((2.0*mu)**(-1.0/(r-1.0)))
            * (Constant(eps) + inner(s, s))**Constant(-0.5*(r-2.0)/(r-1.0)) * s
          - d
        )


class PowerLawVortices(GeneralizedStokesProblem):
    n = 1 # Number of vortices
    mu = 1.0

    def __init__(self, N, r, eps0):
        mesh = UnitSquareMesh(N, N, "crossed")
        constitutive_law = PowerLawFluid(self.mu, r)
        self.u_ex, self.p_ex, self.s_ex, self.f = \
            pStokes_vortices(n=self.n, mu=self.mu, r=r, eps=0.0,
                             degree=6)
        GeneralizedStokesProblem.__init__(self, mesh, constitutive_law,
                                          self.f, eps0)


    def bcs(self, W):
        bc_u = DirichletBC(W.sub(0), (0.0, 0.0), "on_boundary")
        bc_p = DirichletBC(W.sub(1), 0.0, "near(x[0], 0.0) && near(x[1], 0.0)",
                           method="pointwise")
        return [bc_u, bc_p]
