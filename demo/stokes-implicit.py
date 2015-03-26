import dolfin
import matplotlib.pyplot as plt
import numpy as np

from problems.StokesVortices import StokesVortices

mesh_resolutions = [4, 8, 16, 32, 64]
err, est, bnd = [], [], []

for N in mesh_resolutions:
    problem = StokesVortices(N)

    u = problem.solve().split()[0]
    err_1, err_2, err_3 = problem.compute_errors()[:3]
    est_1, est_2, est_3 = problem.estimate_errors()[:3]
    low_1, upp_1 = problem.compute_exact_bounds(problem.u_ex,
                      problem.p_ex, problem.s_ex)[:2]

    dolfin.plot(u, title='solution at N = %d' % N)
    err.append([err_1, err_2, err_3])
    est.append([est_1, est_2, est_3])
    bnd.append([low_1, upp_1])


err = np.array(err, dtype='float')
est = np.array(est, dtype='float')
bnd = np.array(bnd, dtype='float')

for i in xrange(3):
    plt.subplot(3, 1, i+1)
    plt.plot(mesh_resolutions, err[:, i], 'o-')
    plt.plot(mesh_resolutions, est[:, i], 'o-')
    if i == 0:
        plt.plot(mesh_resolutions, bnd[:, 0], 'o-')
        plt.plot(mesh_resolutions, bnd[:, 1], 'o-')
    plt.loglog()

plt.tight_layout()
plt.show(block=True)

dolfin.interactive()
