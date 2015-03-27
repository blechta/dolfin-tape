import dolfin
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from problems.StokesVortices import StokesVortices


# Reduce pivotting of LU solver
dolfin.PETScOptions.set('mat_mumps_cntl_1', 0.001)

mesh_resolutions = [4, 8, 16, 32, 64]
err, est, bnd = [], [], []

comm = dolfin.mpi_comm_world()
prefix = 'results'
f_u = dolfin.XDMFFile(comm, prefix+'/u.xdmf')
f_Err_1 = dolfin.XDMFFile(comm, prefix+'/Err_1.xdmf')
f_Err_2 = dolfin.XDMFFile(comm, prefix+'/Err_2.xdmf')
f_Err_3 = dolfin.XDMFFile(comm, prefix+'/Err_3.xdmf')
f_Est_1 = dolfin.XDMFFile(comm, prefix+'/Est_1.xdmf')
f_Est_2 = dolfin.XDMFFile(comm, prefix+'/Est_2.xdmf')
f_Est_3 = dolfin.XDMFFile(comm, prefix+'/Est_3.xdmf')
f_Low_1 = dolfin.XDMFFile(comm, prefix+'/Low_1.xdmf')
f_Upp_1 = dolfin.XDMFFile(comm, prefix+'/Upp_1.xdmf')

for N in mesh_resolutions:
    problem = StokesVortices(N)

    u = problem.solve().split()[0]
    err_1, err_2, err_3, Err_1, Err_2, Err_3 = problem.compute_errors()
    est_1, est_2, est_3, Est_1, Est_2, Est_3 = problem.estimate_errors()
    low_1, upp_1, Low_1, Upp_1 = problem.compute_exact_bounds(problem.u_ex,
                      problem.p_ex, problem.s_ex)

    dolfin.plot(u, title='solution at N = %d' % N)
    err.append([err_1, err_2, err_3])
    est.append([est_1, est_2, est_3])
    bnd.append([low_1, upp_1])

    f_u << u, N
    f_Err_1 << Err_1, N
    f_Err_2 << Err_2, N
    f_Err_3 << Err_3, N
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
    for i in xrange(3):
        plt.subplot(gs[2*i])
        plt.plot(mesh_resolutions, err[:, i], 'o-', label='err')
        plt.plot(mesh_resolutions, est[:, i], 'o-', label='est')
        if i == 0:
            plt.plot(mesh_resolutions, bnd[:, 0], 'o-', label='lo')
            plt.plot(mesh_resolutions, bnd[:, 1], 'o-', label='up')
        plt.loglog()
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    dolfin.info('Blocking matplotlib figure on rank 0. Close to continue...')
    plt.show()

dolfin.interactive()
