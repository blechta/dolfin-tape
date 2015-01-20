from dolfin import *
import ufl
ufl.set_level(ufl.INFO) # Enable info_{green,red,blue}

import matplotlib.pyplot as plt
import numpy as np

from common import FluxReconstructor


results = ([], [], [], [], [])

for N in [2**i for i in range(2, 7)]:
    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, 'CG', 1)
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v))*dx
    m, n = 1, 1
    u_ex = Expression('sin(m*pi*x[0])*sin(n*pi*x[1])', m=m, n=n, degree=4)
    f = Constant((m*m + n*n)*pi*pi)*u_ex
    L = f*v*dx
    bc = DirichletBC(V, 0.0, lambda x, b: b)
    u = Function(V)
    solve(a == L, u, bc)

    reconstructor = FluxReconstructor(mesh, 1) # TODO: Is it correct degree?
    w = reconstructor.reconstruct(u, f)
    q = Function(w, 0)

    plot(q)

    # TODO: h is not generally constant!
    h = sqrt(2.0)/N
    results[0].append((N, V.dim()))
    energy_error = errornorm(u_ex, u, norm_type='H10')

    # Approximation; works surprisingly well
    err_est = assemble(inner(grad(u)+q, grad(u)+q)*dx)**0.5 \
            + h/pi*assemble(inner(f-div(q), f-div(q))*dx)**0.5
    info_red('Estimator %g, energy_error %g' % (err_est, energy_error))
    results[1].append((err_est, energy_error))

    # Correct way; slow numpy manipulation is used
    DG0 = FunctionSpace(mesh, 'DG', 0)
    v = TestFunction(DG0)
    err0 = assemble(inner(grad(u)+q, grad(u)+q)*v*dx)
    err1 = assemble(inner(f-div(q), f-div(q))*v*dx)
    err0[:] = err0.array()**0.5 + h/pi*err1.array()**0.5
    err_est = err0.norm('l2')
    info_red('Estimator %g, energy_error %g' % (err_est, energy_error))
    results[2].append((err_est, energy_error))

    # Approximation
    err_est = assemble ( ( inner(grad(u)+q, grad(u)+q)**0.5
                         + Constant(h/pi)*inner(f-div(q), f-div(q))**0.5
                         )**2*dx
                       ) ** 0.5
    info_red('Estimator %g, energy_error %g' % (err_est, energy_error))
    results[3].append((err_est, energy_error))

    # Other ways of computing error
    u_err = errornorm(u_ex, u)
    q_ex = Expression(('-m*pi*cos(m*pi*x[0])*sin(n*pi*x[1])',
                       '-n*pi*sin(m*pi*x[0])*cos(n*pi*x[1])'),
                       m=m, n=n, degree=3)
    q_err = errornorm(q_ex, q)
    info_red('u L2-errornorm %g, q L2-errornorm %g)'%(u_err, q_err))
    results[4].append((0.0, q_err))
    Q = w.function_space().sub(0).collapse()
    info_red('||grad(u)+q||_2 = %g'%norm(project(grad(u)+q, Q)))
    info_red('||grad(u)-grad(u_ex)||_2 = %g'%errornorm(q_ex, project(-grad(u), Q)))

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
