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

__all__ = ['StokesVortices']


class NewtonianFluid(object):
    def __init__(self, mu):
        self._mu = mu

    def r(self):
        return 2

    def mu(self):
        return self._mu

    def g(self):
        return lambda s, d, eps: Constant(1.0/(2.0*self._mu))*s - d


class StokesVortices(GeneralizedStokesProblem):
    n = 4 # Number of vortices
    mu = 1.0

    def __init__(self, N):
        mesh = UnitSquareMesh(N, N, "crossed")
        constitutive_law = NewtonianFluid(self.mu)
        self.u_ex, self.p_ex, self.s_ex, self.f = \
            pStokes_vortices(n=self.n, mu=self.mu, r=2, eps=0.0,
                             degree=6)
        GeneralizedStokesProblem.__init__(self, mesh, constitutive_law,
                                          self.f, 0.0)


    def bcs(self, W):
        bc_u = DirichletBC(W.sub(0), (0.0, 0.0), "on_boundary")
        bc_p = DirichletBC(W.sub(1), 0.0, "near(x[0], 0.0) && near(x[1], 0.0)",
                           method="pointwise")
        return [bc_u, bc_p]
