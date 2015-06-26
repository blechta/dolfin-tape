"""This script solves Poisson problem for data with known, manufactured solution
and demonstrates efficiency of an error estimate using equilibrated flux
reconstruction as described in

   [A. Ern, and M. Vohral\'ik, Polynomial-degree-robust a posteriori estimates
   in a unified setting for conforming, nonconforming, discontinuous Galerkin,
   and mixed discretizations, SIAM J. Numer. Anal., 53 (2015), pp. 1058-1081.]
"""

from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

from common import FluxReconstructor
from common.cell_diameter import CellDiameters


results = []

for N in [2**i for i in range(2, 7)]:
    # Manufacured solution
    m, n = 5, 3
    u_ex = Expression('sin(m*pi*x[0])*sin(n*pi*x[1])', m=m, n=n, degree=4)
    f = Constant((m*m + n*n)*pi*pi)*u_ex

    # Solve Poisson problem
    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, 'CG', 1)
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v))*dx
    L = f*v*dx
    bc = DirichletBC(V, 0.0, lambda x, b: b)
    u = Function(V)
    solve(a == L, u, bc)
    plot(u, title='u (%d x %d)' % (N, N))

    # Reconstruct flux q in H(div)
    reconstructor = FluxReconstructor(mesh, 1)
    tic()
    w = reconstructor.reconstruct(grad(u), f)
    t_flux_reconstructor = toc()
    info_green('Flux reconstruction timing: %g' % t_flux_reconstructor)
    q = Function(w, 0)
    plot(q, title='q (%d x %d)' % (N, N))

    # Compute actual error using known solution
    energy_error = errornorm(u_ex, u, norm_type='H10')

    # Compute error estimate using equilibrated stress reconstruction
    h = CellDiameters(mesh)
    DG0 = FunctionSpace(mesh, 'DG', 0)
    v = TestFunction(DG0)
    err0 = assemble(inner(grad(u)+q, grad(u)+q)*v*dx)
    err1 = assemble((h/pi)**2*inner(f-div(q), f-div(q))*v*dx)
    err0[:] = err0.array()**0.5 + err1.array()**0.5
    err_est = err0.norm('l2')

    # Issue information and store results
    info_red('Estimator %g, energy_error %g' % (err_est, energy_error))
    results.append((N, V.dim(), err_est, energy_error, t_flux_reconstructor))

results = np.array(results, dtype='float')

def slope(x0, y0, power):
    return lambda x: y0*(x/x0)**power

# Plot comparison of errors and estimates
plt.subplot(2, 1, 1)
plt.plot(results[:, 0], results[:, 2], 'o-', label='estimate')
plt.plot(results[:, 0], results[:, 3], 'o-', label='error')
plt.plot(results[:, 0],
         map(slope(results[0, 0], 3.0*results[0, 2], -1.0), results[:, 0]),
         'k--', label=r'$C\,h$')
plt.title('Error and its estimate')
plt.xlabel(r'$1/h$')
plt.ylabel(r'$||\nabla u-\nabla u_h||_2$')
plt.loglog()
plt.legend()

# Plot timing of flux reconstructions
plt.subplot(2, 1, 2)
plt.plot(results[:, 1], results[:, 4], 'o-')
plt.plot(results[:, 1],
         map(slope(results[0, 1], 3.0*results[0, 4], 1.0), results[:, 1]),
         'k--', label='linear scaling')
plt.title('Flux reconstructor timing')
plt.xlabel('#DOFs')
plt.ylabel('t [s]')
plt.loglog()
plt.legend()

plt.tight_layout()
plt.show(block=True)

interactive()
