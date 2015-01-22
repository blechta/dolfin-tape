from dolfin import *
import ufl
ufl.set_level(ufl.INFO) # Enable info_{green,red,blue}

import matplotlib.pyplot as plt
import numpy as np

from common import FluxReconstructor


results = ([], [], [], [], [])

for N in [2**i for i in range(2, 7)]:
    mesh = UnitSquareMesh(N, N, 'crossed')
    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    W = V*Q
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    a = ( inner(grad(u), grad(v)) - p*div(v) - q*div(u) )*dx
    u_ex = Expression(('+pow(sin(n*pi*x[0]), 2) * sin(2.0*n*pi*x[1])',
                       '-pow(sin(n*pi*x[1]), 2) * sin(2.0*n*pi*x[0])'),
                      n=4.0, degree=6)
    p_ex = Expression('0.0')
    f = Expression(('+2.0*n*n*pi*pi*( 2.0*pow(sin(n*pi*x[0]), 2) - cos(2.0*n*pi*x[0]) ) * sin(2.0*n*pi*x[1])',
                    '-2.0*n*n*pi*pi*( 2.0*pow(sin(n*pi*x[1]), 2) - cos(2.0*n*pi*x[1]) ) * sin(2.0*n*pi*x[0])'),
                   n=4.0, degree=6)
    L = inner(f, v)*dx
    bc_u = DirichletBC(W.sub(0), (0.0, 0.0), lambda x, b: b)
    bc_p = DirichletBC(W.sub(1), 0.0, "near(x[0], 0.0) && near(x[1], 0.0)", method="pointwise")
    w = Function(W)
    solve(a == L, w, [bc_u, bc_p])

    u, p = w.split()

    print errornorm(u_ex, u, norm_type='H10'), errornorm(Expression('0.0'), p)

    plot(u)

    gdim = mesh.geometry().dim()
    reconstructor = FluxReconstructor(mesh, 1) # TODO: Is it correct degree?
    w = [reconstructor.reconstruct(-u[i], -f[i] + p.dx(i)) for i in range(gdim)]
    q = as_vector([Function(w[i], 0) for i in range(gdim)])
    #q = [Function(w[i], 0) for i in range(gdim)]

    for i in range(gdim):
        plot(q[i])

    # TODO: We are missing everywhere pressure error!

    Au = 0.5*(sqrt(5.0)+1.0)
    # TODO: h is not generally constant!
    h = sqrt(2.0)/N
    # TODO: Obtain Brezzi constant by eigensolver
    beta = 0.1

    results[0].append((N, V.dim()))
    energy_error = errornorm(u_ex, u, norm_type='H10') \
                 + errornorm(p_ex, p)

    # Approximation; works surprisingly well
    err_est = assemble(inner(grad(u)-q, grad(u)-q)*dx)**0.5 \
            + h/pi*assemble(inner(div(q)-grad(p)+f, div(q)-grad(p)+f)*dx)**0.5 \
            + 1.0/beta*assemble(div(u)*div(u)*dx)**0.5
    #err_est *= Au
    info_red('Estimator %g, energy_error %g' % (err_est, energy_error))
    results[1].append((err_est, energy_error))

    # Correct way; slow numpy manipulation is used
    DG0 = FunctionSpace(mesh, 'DG', 0)
    v = TestFunction(DG0)
    err0 = assemble(inner(grad(u)-q, grad(u)-q)*v*dx)
    err1 = assemble(inner(div(q)-grad(p)+f, div(q)-grad(p)+f)*v*dx)
    err2 = assemble(div(u)*div(u)*v*dx)
    err0[:] = err0.array()**0.5 + h/pi*err1.array()**0.5
    err_est = err0.norm('l2') + 1.0/beta*err2.norm('l2')
    #err_est *= Au
    info_red('Estimator %g, energy_error %g' % (err_est, energy_error))
    results[2].append((err_est, energy_error))

    # Approximation
    err_est = assemble ( ( inner(grad(u)-q, grad(u)-q)**0.5
                         + Constant(h/pi)*inner(div(q)-grad(p)+f, div(q)-grad(p)+f)**0.5
                         )**2*dx
                         + div(u)*div(u)*dx
                       ) ** 0.5
    #err_est *= Au
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
