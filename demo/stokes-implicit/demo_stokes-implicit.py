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

TODO: This implements Newtonian fluid. Prepare non-Newtonian.
"""

import dolfin
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from dolfintape.demo_problems import StokesVortices, PowerLawVortices


# UFLACS form compiler performs much better for complex forms
dolfin.parameters['form_compiler']['representation'] = 'uflacs'

# Reduce pivotting of LU solver
dolfin.PETScOptions.set('mat_mumps_cntl_1', 0.001)

eps = 10.0
mesh_resolutions = [4, 8, 16, 32, 64]
err, est, bnd = [], [], []

comm = dolfin.mpi_comm_world()
prefix = 'results'
f_u = dolfin.XDMFFile(comm, prefix+'/u.xdmf')
f_Est_1 = dolfin.XDMFFile(comm, prefix+'/Est_1.xdmf')
f_Est_2 = dolfin.XDMFFile(comm, prefix+'/Est_2.xdmf')
f_Est_3 = dolfin.XDMFFile(comm, prefix+'/Est_3.xdmf')
f_Low_1 = dolfin.XDMFFile(comm, prefix+'/Low_1.xdmf')
f_Upp_1 = dolfin.XDMFFile(comm, prefix+'/Upp_1.xdmf')


for N in mesh_resolutions:
    #problem = StokesVortices(N)
    problem = PowerLawVortices(N, 2.5, eps)

    u = problem.solve_adaptive_eps().split()[0]
    eps = problem._eps
    err_1 = problem.compute_errors()[:1]
    est_1, est_2, est_3, Est_1, Est_2, Est_3 = \
            problem.estimate_errors_overall()
    low_1, upp_1, Low_1, Upp_1 = problem.compute_exact_bounds(problem.u_ex,
                      problem.p_ex, problem.s_ex)

    dolfin.plot(u, title='solution at N = %d' % N)
    err.append([err_1])
    est.append([est_1, est_2, est_3])
    bnd.append([low_1, upp_1])

    u.rename("velocity", "")
    Est_1.rename("residual_1 estimate", "")
    Est_2.rename("residual_2 estimate", "")
    Est_3.rename("residual_3 estimate", "")
    Low_1.rename("residual_1 lower estimate", "")
    Upp_1.rename("residual_1 upper estimate", "")

    # FIXME: Deadlock in parallel! Why?
    if dolfin.MPI.size(comm) == 1:
        f_u << u, N
        f_Est_1 << Est_1, N
        f_Est_2 << Est_2, N
        f_Est_3 << Est_3, N
        f_Low_1 << Low_1, N
        f_Upp_1 << Upp_1, N


if dolfin.MPI.rank(dolfin.mpi_comm_world()) == 0:
    err = np.array(err, dtype='float')
    est = np.array(est, dtype='float')
    bnd = np.array(bnd, dtype='float')

    gs = gridspec.GridSpec(3, 2, width_ratios=[7, 1])

    plt.subplot(gs[0])
    plt.plot(mesh_resolutions, err[:, 0], 'o-', label='lft')
    plt.plot(mesh_resolutions, est[:, 0], 'o-', label='est')
    plt.plot(mesh_resolutions, bnd[:, 0], 'o-', label='lo')
    plt.plot(mesh_resolutions, bnd[:, 1], 'o-', label='up')
    plt.loglog()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.subplot(gs[2])
    plt.plot(mesh_resolutions, est[:, 1], 'o-', label='est')
    plt.loglog()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.subplot(gs[4])
    plt.plot(mesh_resolutions, est[:, 2], 'o-', label='est')
    plt.loglog()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.tight_layout()
    dolfin.info('Blocking matplotlib figure on rank 0. Close to continue...')
    plt.savefig(prefix+'/convergence.pdf')
    plt.show()

dolfin.interactive()

dolfin.list_timings(dolfin.TimingClear_keep, [dolfin.TimingType_wall])
