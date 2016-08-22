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

"""This script solves generalized Stokes problem in the framework of implicit
constitutive theory for data with known, manufactured solution and
demonstrates efficiency of an error estimate using equilibrated stress
reconstruction as described in

    [J. Blechta, J. M\'alek, M. Vohral\'ik, Generalized Stokes flows of
    implicitly constituted fluids: a posteriori error control and full
    adaptivity, in preparation, 2016.]
"""

import dolfin
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from dolfintape.demo_problems.GeneralizedStokes import GeneralizedStokesProblem
from dolfintape.demo_problems.PowerLawVortices import PowerLawFluid
from dolfintape.demo_problems.exact_solutions import pStokes_vortices


class DolfinObstacleProblem(GeneralizedStokesProblem):
    mu = 1.0
    eps0 = 100.0
    #f = dolfin.Constant((0.0, 0.0))

    def __init__(self, r):
        #self._mesh, self._ff = self.load_mesh()
        self._mesh = self.load_mesh()
        _, _, _, self.f = pStokes_vortices(n=1, mu=self.mu, r=r, eps=0.0,
                                           degree=1)
        self._constitutive_law = PowerLawFluid(self.mu, r)
        GeneralizedStokesProblem.__init__(self, self._mesh,
                                          self._constitutive_law,
                                          self.f, self.eps0)

    def load_mesh(self):
        #mesh = dolfin.Mesh('dolfin_fine.xml.gz')
        mesh = dolfin.Mesh('dolfin_coarse.xml.gz')
        #ff = dolfin.MeshFunction('size_t', mesh,
        #                         'dolfin_fine_subdomains.xml.gz')
        #return mesh, ff
        return mesh

    def bcs(self, W):
        #bc_u0 = dolfin.DirichletBC(W.sub(0), ( 0.0, 0.0), self._ff, 0)
        #bc_u1 = dolfin.DirichletBC(W.sub(0), (-1.0, 0.0), self._ff, 1)
        bc_u = dolfin.DirichletBC(W.sub(0), (0.0, 0.0), "on_boundary")
        bc_p = dolfin.DirichletBC(W.sub(1), 0.0,
                                  "near(x[0], 0.0) && near(x[1], 0.0)",
                                  method="pointwise")
        #return [bc_u0, bc_u1, bc_p]
        return [bc_u, bc_p]

    def criterion_h(self):
        est_1, est_2, est_3, Est_1, Est_2, Est_3 = \
                self.estimate_errors_overall()
        u = self._w.sub(0)
        N = self._w.function_space().dim()

        dolfin.plot(u, title='solution at N = %d' % N)
        dofs.append(N)
        est.append([est_1, est_2, est_3])

        u.rename("velocity", "")
        Est_1.rename("residual_1 estimate", "")
        Est_2.rename("residual_2 estimate", "")
        Est_3.rename("residual_3 estimate", "")

        f_u.write(u, N)
        f_Est_1.write(Est_1, N)
        f_Est_2.write(Est_2, N)
        f_Est_3.write(Est_3, N)

        return GeneralizedStokesProblem.criterion_h(self)


# UFLACS form compiler performs much better for complex forms
dolfin.parameters['form_compiler']['representation'] = 'uflacs'

# Reduce pivotting of LU solver
dolfin.PETScOptions.set('mat_mumps_cntl_1', 0.001)

dofs, est = [], []

comm = dolfin.mpi_comm_world()
prefix = 'results'
f_u = dolfin.XDMFFile(comm, prefix+'/u.xdmf')
f_Est_1 = dolfin.XDMFFile(comm, prefix+'/Est_1.xdmf')
f_Est_2 = dolfin.XDMFFile(comm, prefix+'/Est_2.xdmf')
f_Est_3 = dolfin.XDMFFile(comm, prefix+'/Est_3.xdmf')

problem = DolfinObstacleProblem(2.5)
problem.solve_adaptive_h()

if dolfin.MPI.rank(dolfin.mpi_comm_world()) == 0:
    est = np.array(est, dtype='float')

    gs = gridspec.GridSpec(3, 2, width_ratios=[7, 1])

    plt.subplot(gs[0])
    plt.plot(dofs, est[:, 0], 'o-', label='est')
    plt.loglog()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.subplot(gs[2])
    plt.plot(dofs, est[:, 1], 'o-', label='est')
    plt.loglog()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.subplot(gs[4])
    plt.plot(dofs, est[:, 2], 'o-', label='est')
    plt.loglog()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.tight_layout()
    dolfin.info('Blocking matplotlib figure on rank 0. Close to continue...')
    plt.savefig(prefix+'/convergence.pdf')
    plt.show()

dolfin.interactive()

dolfin.list_timings(dolfin.TimingClear_keep, [dolfin.TimingType_wall])
