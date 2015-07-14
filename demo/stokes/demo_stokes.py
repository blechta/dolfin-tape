"""This script solves Stokes problem for data with known, manufactured solution
and demonstrates efficiency of an error estimate using equilibrated stress
reconstruction in a spirit similar to

    [A. Hannukainen, R. Stenberg, and M. Vohral\'ik, A unified framework for
    a posteriori error estimation for the Stokes problem, Numer. Math., 122
    (2012), pp. 725-769.]

TODO: is there a paper with more closer approach? Hannukainen et al. uses dual
      mesh approach instead of more recent reconstruction on patches.
"""

from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

from dolfintape import FluxReconstructor, CellDiameters
from dolfintape.demo_problems import pStokes_vortices


results = []

for N in [2**i for i in xrange(2, 7)]:
    mesh = UnitSquareMesh(N, N, 'crossed')

    # Retrieve manufactured solution
    u_ex, p_ex, _, f = pStokes_vortices(n=4, mu=1.0, r=2, eps=0.0,
                                        degree=6, domain=mesh)

    # Solve Stokes problem
    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    W = V*Q
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    a = ( inner(grad(u), grad(v)) - p*div(v) - q*div(u) )*dx
    L = inner(f, v)*dx
    bc_u = DirichletBC(W.sub(0), (0.0, 0.0), lambda x, b: b)
    bc_p = DirichletBC(W.sub(1), 0.0, "near(x[0], 0.0) && near(x[1], 0.0)",
                       method="pointwise")
    w = Function(W)
    solve(a == L, w, [bc_u, bc_p])
    u, p = w.split(deepcopy=True)
    pt = plot(u, title='u (%d x %d)' % (N, N))
    pt.write_png('results/stokes_u_%d_%d' % (N, N))

    # Adjust presure to zero mean
    p.vector()[:] -= assemble(p*dx)/assemble(Constant(1.0)*dx(mesh))

    # Reconstruct extra stress q in H(div)
    reconstructor = FluxReconstructor(mesh, 2)
    q = []
    tic()
    for i in range(mesh.geometry().dim()):
        w = reconstructor.reconstruct(-grad(u[i]), -f[i] + p.dx(i))
        q.append(w.sub(0, deepcopy=True))
    t_flux_reconstructor = toc()
    info_green('Flux reconstruction timing: %g' % t_flux_reconstructor)
    q = as_vector(q)
    for i in range(mesh.geometry().dim()):
        pt = plot(q[i], title='q[%d] (%d x %d)' % (i, N, N))
        pt.write_png('results/stokes_q%d_%d_%d' % (i, N, N))

    # Cell size and inf-sup constant
    h = CellDiameters(mesh)
    beta = (0.5 - pi**-1)**0.5 # upper bound on inf-sup constant on square

    # Compute actual error using known solution
    energy_error = ( errornorm(u_ex, u, norm_type='H10')**2
                   + beta**2*errornorm(p_ex, p)**2 )**0.5

    # Compute error estimate using equilibrated stress reconstruction
    DG0 = FunctionSpace(mesh, 'DG', 0)
    v = TestFunction(DG0)
    err0 = assemble(inner(grad(u)-q, grad(u)-q)*v*dx)
    err1 = assemble((h/pi)**2*inner(div(q)-grad(p)+f, div(q)-grad(p)+f)*v*dx)
    err2 = assemble(div(u)*div(u)*v*dx)
    err0[:] = err0.array()**0.5 + err1.array()**0.5
    err_est = ( err0.norm('l2')**2 + (1.0/beta*err2.norm('l2'))**2)**0.5

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
         map(slope(results[0, 0], 3.0*results[0, 2], -2.0), results[:, 0]),
         'k--', label=r'$C\,h^2$')
plt.title('Error and its estimate')
plt.xlabel(r'$1/h$')
plt.ylabel(r'$(||\mathbf{\nabla u-\nabla u}_h||_2^2'
           r'+\beta^2||p-p_h||_2^2)^{1/2}$')
plt.loglog()
plt.legend(loc=3)

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
plt.legend(loc=4)

plt.tight_layout()
plt.savefig('results/convergence.pdf')
plt.show(block=True)

interactive()
