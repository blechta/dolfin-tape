from dolfin import *
import ufl
ufl.set_level(ufl.INFO) # Enable info_{green,red,blue}

import matplotlib.pyplot as plt
import numpy as np

from common import FluxReconstructor
from common.deviatoric_space import TensorFunctionSpace, deviatoric


class ImplicitStokesFluxReconstructor(FluxReconstructor):

    def __init__(self, mesh, degree, r, mu):
        self._r = Constant(r)
        self._mu = Constant(mu)
        FluxReconstructor.__init__(self, mesh, degree)

    def L(self, p, stress_component, force_component, residual_component):
        """rhs of (6.7); one component"""
        v, phi = TestFunctions(self._W)
        hat = self._hat
        mu, r = float(self._mu), float(self._r)
        weight = Constant((2.0*mu)**(-1.0/r))
        I_RTN = lambda x: x # Dummy interpolator
        return ( weight*I_RTN(hat[p]*inner(stress_component, v))
                 -hat[p]*force_component*phi
                 +inner(grad(hat[p]), stress_component)*phi
                 +residual_component*phi )*self.dp(p)

    def a(self, p):
        """lhs of (6.7); one component"""
        d, q = TrialFunctions(self._W)
        v, phi = TestFunctions(self._W)
        mu, r = float(self._mu), float(self._r)
        weight = Constant((2.0*mu)**(-1.0/r))
        return ( weight*inner(d, v) + inner(q, div(v)) + inner(phi, div(d)) )*self.dp(p)


results = ([], [], [], [], [])

for N in [2**i for i in xrange(2, 7)]:
    mesh = UnitSquareMesh(N, N, 'crossed')
    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    S = TensorFunctionSpace(mesh, 'DG', 1, symmetry=True, zero_trace=True)
    W = MixedFunctionSpace([V, Q, S])

    w = Function(W)
    (u, p, s) = split(w)
    (v, q, t) = TestFunctions(W)
    s = deviatoric(s)
    t = deviatoric(t)
    d = sym(grad(u))

    mu = Constant(1.0)
    u_ex = Expression(('+pow(sin(n*pi*x[0]), 2) * sin(2.0*n*pi*x[1])',
                       '-pow(sin(n*pi*x[1]), 2) * sin(2.0*n*pi*x[0])'),
                      n=4.0, degree=6)
    p_ex = Expression('0.0')
    s_ex = 2.0*mu*Expression(
            (('2.0*n*pi*sin(2.0*n*pi*x[0])*sin(2.0*n*pi*x[1])',
              '2.0*n*pi*( -pow(sin(n*pi*x[0]), 2)*pow(sin(n*pi*x[1]), 2) '
                        ' + 0.5*pow(cos(n*pi*x[0]), 2)*pow(sin(n*pi*x[1]), 2) '
                        ' + 0.5*pow(cos(n*pi*x[0]), 2)*pow(sin(n*pi*x[1]), 2) )'),
             ('2.0*n*pi*sin(2.0*n*pi*x[0])*sin(2.0*n*pi*x[1])',
              '2.0*n*pi*( -pow(sin(n*pi*x[0]), 2)*pow(sin(n*pi*x[1]), 2) '
                        ' + 0.5*pow(cos(n*pi*x[0]), 2)*pow(sin(n*pi*x[1]), 2) '
                        ' + 0.5*pow(cos(n*pi*x[0]), 2)*pow(sin(n*pi*x[1]), 2) )')),
             n=4.0, degree=6)

    f = Expression(('+2.0*n*n*pi*pi*( 2.0*pow(sin(n*pi*x[0]), 2) - cos(2.0*n*pi*x[0]) ) * sin(2.0*n*pi*x[1])',
                    '-2.0*n*n*pi*pi*( 2.0*pow(sin(n*pi*x[1]), 2) - cos(2.0*n*pi*x[1]) ) * sin(2.0*n*pi*x[0])'),
                   n=4.0, degree=6)
    g = s-2.0*mu*d
    F = ( inner(s, grad(v)) - p*div(v) - q*div(u) + inner(g, t) - inner(f, v) )*dx
    F += Constant(0.0)*p*q*dx # Add some diagonal zeros to make PETSc happy
    bc_u = DirichletBC(W.sub(0), (0.0, 0.0), lambda x, b: b)
    bc_p = DirichletBC(W.sub(1), 0.0, "near(x[0], 0.0) && near(x[1], 0.0)", method="pointwise")
    solve(F == 0, w, [bc_u, bc_p])

    u, p, s = w.split()
    s = deviatoric(s)

    print errornorm(u_ex, u, norm_type='H10'), errornorm(Expression('0.0'), p)

    plot(u)

    gdim = mesh.geometry().dim()
    reconstructor = ImplicitStokesFluxReconstructor(mesh, 2, 2.0, mu)
    q = []
    I = Identity(gdim)
    stress = -p*I + s
    for i in xrange(gdim):
        w = reconstructor.reconstruct(stress[i, :], f[i], 0.0)
        q.append(w.sub(0, deepcopy=True))
    q = as_vector(q)

    for i in xrange(gdim):
        plot(q[i])

    # TODO: We are missing everywhere pressure error!

    # TODO: h is not generally constant!
    h = sqrt(2.0)/N
    # TODO: Obtain Brezzi constants by eigensolver
    beta = Constant(0.1)
    gamma = Constant(0.1)

    results[0].append((N, V.dim()))
    # FIXME: Correct error estimates computed from exact solution! Add residuals
    #        from the paper!
    energy_error = errornorm(u_ex, u, norm_type='H10') \
                 + errornorm(p_ex, p)

    # Approximation; works surprisingly well
    err_est = assemble(inner(s-p*I-q, s-p*I-q)*dx)**0.5 \
            + h/pi*assemble(inner(div(q)+f, div(q)+f)*dx)**0.5 \
            #+ 1.0/beta*assemble(div(u)*div(u)*dx)**0.5 \
            #+ 1.0/gamma*assemble(inner(g, g)*dx)**0.5
    info_red('Estimator %g, energy_error %g' % (err_est, energy_error))
    results[1].append((err_est, energy_error))

    # Correct way; slow numpy manipulation is used
    DG0 = FunctionSpace(mesh, 'DG', 0)
    v = TestFunction(DG0)
    err0 = assemble(inner(s-p*I-q, s-p*I-q)*v*dx)
    err1 = assemble(inner(div(q)+f, div(q)+f)*v*dx)
    err2 = assemble(div(u)*div(u)*v*dx)
    err3 = assemble(inner(g, g)*dx)**0.5
    err0[:] = err0.array()**0.5 + h/pi*err1.array()**0.5
    err_est = err0.norm('l2') #+ 1.0/beta*err2.norm('l2') + 1.0/gamma*err3.norm('l2')
    info_red('Estimator %g, energy_error %g' % (err_est, energy_error))
    results[2].append((err_est, energy_error))

    # Approximation
    err_est = assemble ( ( inner(s-p*I-q, s-p*I-q)**0.5
                         + Constant(h/pi)*inner(div(q)+f, div(q)+f)**0.5
                         )**2*dx
                         #+ 1.0/beta*div(u)*div(u)*dx
                         #+ 1.0/gamma*inner(g, g)*dx
                       ) ** 0.5
    info_red('Estimator %g, energy_error %g' % (err_est, energy_error))
    results[3].append((err_est, energy_error))

    # Other ways of computing error
    u_err = errornorm(u_ex, u)
    #q_ex = Expression(('-m*pi*cos(m*pi*x[0])*sin(n*pi*x[1])',
    #                   '-n*pi*sin(m*pi*x[0])*cos(n*pi*x[1])'),
    #                   m=m, n=n, degree=3)
    #q_err = errornorm(q_ex, q)
    #info_red('u L2-errornorm %g, q L2-errornorm %g)'%(u_err, q_err))
    #results[4].append((0.0, q_err))
    results[4].append((0.0, 1.0))
    #Q = w.function_space().sub(0).collapse()
    #info_red('||grad(u)+q||_2 = %g'%norm(project(grad(u)+q, Q)))
    #info_red('||grad(u)-grad(u_ex)||_2 = %g'%errornorm(q_ex, project(-grad(u), Q)))

results = np.array(results, dtype='float')

plt.subplot(4, 1, 1)
plt.plot(results[0, :, 0], results[1, :, 0], 'o-')
plt.plot(results[0, :, 0], results[1, :, 1], 'o-')
plt.loglog()
plt.subplot(4, 1, 2)
plt.plot(results[0, :, 0], results[2, :, 0], 'o-')
plt.plot(results[0, :, 0], results[2, :, 1], 'o-')
plt.loglog()
plt.subplot(4, 1, 3)
plt.plot(results[0, :, 0], results[3, :, 0], 'o-')
plt.plot(results[0, :, 0], results[3, :, 1], 'o-')
plt.loglog()
plt.subplot(4, 1, 4)
plt.plot(results[0, :, 0], results[4, :, 0], 'o-')
plt.plot(results[0, :, 0], results[4, :, 1], 'o-')
plt.loglog()

plt.tight_layout()
plt.show(block=True)

interactive()
