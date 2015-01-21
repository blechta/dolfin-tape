from dolfin import *
import ufl
ufl.set_level(ufl.INFO) # Enable info_{green,red,blue}

import matplotlib.pyplot as plt
import numpy as np

from common import FluxReconstructor


results = ([], [], [], [], [])

for N in [2**i for i in range(2, 8)]:
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
    # FIXME: It's wrong; we need to supply different linear form to FluxReconstructor
    w = [reconstructor.reconstruct(-u[i], -f[i] + p.dx(i)) for i in range(gdim)]
    q = as_vector([Function(w[i], 0) for i in range(gdim)])
    #q = [Function(w[i], 0) for i in range(gdim)]

    for i in range(gdim):
        plot(q[i])
    interactive()
