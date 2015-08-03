from dolfin import *
import ufl

from dolfintape import FluxReconstructor, CellDiameters

#parameters['form_compiler']['representation'] = 'uflacs'

mesh = UnitSquareMesh(10, 10, 'crossed')
V = FunctionSpace(mesh, 'Lagrange', 1)
f = Expression("1.+cos(2.*pi*x[0])*sin(2.*pi*x[1])", domain=mesh, degree=2)


class ReconstructorCache(dict):
    def __setitem__(self, key, value):
        mesh, degree = key
        dict.__setitem__(self, (mesh.id(), degree), value)
    def __getitem__(self, key):
        mesh, degree = key
        try:
            return dict.__getitem__(self, (mesh.id(), degree))
        except KeyError:
            self[key] = reconstructor = FluxReconstructor(mesh, degree)
            return reconstructor

reconstructor_cache = ReconstructorCache()


def solve_p_laplace(p, eps, V, rhs, u0=None):
    p1 = p/(p-1)
    p = Constant(p)
    p1 = Constant(p1)
    eps = Constant(eps)

    mesh = V.mesh()
    dx = Measure('dx', domain=mesh)

    # Initial approximation for Newton
    u = u0.copy(deepcopy=True) if u0 else Function(V)

    # Integrable right-hand side
    if isinstance(rhs, GenericFunction):
        L, f = rhs*TestFunction(V)*dx, rhs
    # Generic functional
    else:
        assert isinstance(rhs, ufl.Form)
        assert len(rhs.arguments()) == 1
        import pdb; pdb.set_trace()
        rhs = ufl.replace(rhs, {rhs.arguments()[0]: TestFunction(V)})
        L, f = rhs, None

    # Problem formulation
    E = 1./p*(eps + dot(grad(u), grad(u)))**(0.5*p)*dx - action(L, u)
    F = derivative(E, u)
    bc = DirichletBC(V, 0.0, lambda x, onb: onb)
    solve(F == 0, u, bc)

    # Skip error estimation if rhs is not integrable
    if not f:
        return u, None, None, None

    # Reconstruct flux q in H^p1(div) s.t.
    #       q ~ -S
    #   div q ~ f
    S = inner(grad(u), grad(u))**(0.5*p-1.0) * grad(u)
    S_eps = (eps + inner(grad(u), grad(u)))**(0.5*p-1.0) * grad(u)
    tic()
    reconstructor = reconstructor_cache[(V.mesh(), V.ufl_element().degree())]
    q = reconstructor.reconstruct(S, f).sub(0, deepcopy=False)
    info_green('Flux reconstruction timing: %g seconds' % toc())

    DG0 = FunctionSpace(mesh, 'Discontinuous Lagrange', 0)

    # Compute error estimate using equilibrated stress reconstruction
    v = TestFunction(DG0)
    h = CellDiameters(mesh)
    Cp = Constant(2.0*(0.5*p)**(1.0/p)) # Poincare estimates by [Chua, Wheeden 2006]
    est0 = assemble(((Cp*h*(f-div(q)))**2)**(0.5*p1)*v*dx)
    est1 = assemble(inner(S_eps+q, S_eps+q)**(0.5*p1)*v*dx)
    est2 = assemble(inner(S_eps-S, S_eps-S)**(0.5*p1)*v*dx)
    p1 = float(p1)
    est_h   = est0.array()**(1.0/p1) + est1.array()**(1.0/p1)
    est_eps = est2.array()**(1.0/p1)
    est_tot = est_h + est_eps
    Est_h   = MPI.sum( mesh.mpi_comm(), (est_h  **p1).sum() )**(1.0/p1)
    Est_eps = MPI.sum( mesh.mpi_comm(), (est_eps**p1).sum() )**(1.0/p1)
    Est_tot = MPI.sum( mesh.mpi_comm(), (est_tot**p1).sum() )**(1.0/p1)
    info_red('Error estimates: overall %g, discretization %g, regularization %g'
             % (Est_tot, Est_h, Est_eps))

    return u, Est_h, Est_eps, Est_tot


def solve_problem(p, epsilons, zero_guess=False):
    p1 = p/(p-1) # Dual Lebesgue exponent
    p = Constant(p)

    u = None
    for eps in epsilons:
        u = solve_p_laplace(p, eps, V, f, None if zero_guess else u)[0]

    # Define residal form
    E = 1./p*(eps + dot(grad(u), grad(u)))**(0.5*p)*dx - f*u*dx
    R = derivative(E, u)

    # global lifting of residual
    V_high = FunctionSpace(mesh, 'Lagrange', 4)
    r_glob = None
    for eps in epsilons[:4]:
        parameters['form_compiler']['quadrature_degree'] = 4
        r_glob = solve_p_laplace(p, eps, V_high, R)[0]
        parameters['form_compiler']['quadrature_degree'] = -1
        plot(r_glob)
        r_norm_glob = sobolev_norm(r_glob, p)**(p/p1)
        info("||r|| = %g" % r_norm_glob)
    interactive()

    # local lifting of residual
    r_norm_loc = 0.0
    cf = CellFunction('size_t', mesh)
    r_loc = Function(V)
    r_loc_temp = Function(V)
    for v in vertices(mesh):
        cf.set_all(0)
        for c in cells(v):
            cf[c] = 1
        submesh = SubMesh(mesh, cf, 1)
        V_loc = FunctionSpace(submesh, 'Lagrange', 4)
        E = ( 1./p*(eps + dot(grad(u), grad(u)))**(0.5*p) - f*u ) * dx(submesh)
        R = derivative(E, u, TestFunction(V_loc))
        for eps in epsilons[:4]:
            parameters['form_compiler']['quadrature_degree'] = 4
            r = solve_p_laplace(p, eps, V_loc, R)[0]
            parameters['form_compiler']['quadrature_degree'] = -1
            #plot(r)
        r_norm_loc += sobolev_norm(r, p)**p
        r.set_allow_extrapolation(True)
        r_loc_temp.interpolate(r)
        r_loc_vec = r_loc.vector()
        r_loc_vec += r_loc_temp.vector()
    r_norm_loc **= 1.0/p1
    plot(r_loc)

    info("||r|| = %g, \sum_a ||r_a|| = %g" % (r_norm_glob, r_norm_loc))
    interactive()

def sobolev_norm(u, p):
    p = Constant(p) if p is not 2 else p
    return assemble(inner(grad(u), grad(u))**(p/2)*dx)**(1.0/p)


if __name__ == '__main__':
    import numpy as np

    # p = 11.0
    solve_problem(11.0, [10.0**i for i in np.arange(1.0,  -6.0, -0.5)])

    # p = 1.1; works better with zero guess
    solve_problem( 1.1, [10.0**i for i in np.arange(0.0, -22.0, -2.0)], zero_guess=True)
