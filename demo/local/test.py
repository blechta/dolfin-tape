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

"""This script finds an approximation of p-Laplace problem and then uses
its residual in W^{-1,q} to demonstrate localization result of

    [J. Blechta, J. M\'alek, M. Vohral\'ik, Localization of $W^{-1,q}$
    norm for local a posteriori efficiency, in preparation, 2016.]
"""

from dolfin import *
import ufl
import numpy as np

from dolfintape import FluxReconstructor, CellDiameters
from dolfintape.poincare import poincare_friedrichs_cutoff
from dolfintape.hat_function import hat_function

not_working_in_parallel('This')


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


def solve_p_laplace(p, eps, V, f, df, u0=None, exact_solution=None):
    """Approximate (p, eps)-Laplace problem with rhs of the form

        (f, v) - (df, grad(v))  with test function v

    on space V with initial Newton approximation u0. Return solution
    approximation, discretization err estimate, regularization err
    estimate, total err estimate and energy-like upper estimate (if
    exact solution is provided).
    """
    p1 = p/(p-1)
    p = Constant(p)
    p1 = Constant(p1)
    eps = Constant(eps)

    mesh = V.mesh()
    dx = Measure('dx', domain=mesh)

    # Initial approximation for Newton
    u = u0.copy(deepcopy=True) if u0 else Function(V)

    # Problem formulation
    S = inner(grad(u), grad(u))**(0.5*p-1.0) * grad(u) + df
    S_eps = (eps + inner(grad(u), grad(u)))**(0.5*p-1.0) * grad(u) + df
    v = TestFunction(V)
    F_eps = ( inner(S_eps, grad(v)) - f*v ) * dx
    bc = DirichletBC(V, exact_solution if exact_solution else 0.0,
                     "on_boundary")
    solve(F_eps == 0, u, bc,
          solver_parameters={'newton_solver':
                                {'maximum_iterations': 50,
                                #{'maximum_iterations': 500,
                                 'report': True}
                            })

    # Reconstruct flux q in H^p1(div) s.t.
    #       q ~ -S
    #   div q ~ f
    tic()
    reconstructor = reconstructor_cache[(V.mesh(), V.ufl_element().degree())]
    q = reconstructor.reconstruct(S, f).sub(0, deepcopy=False)
    info_green('Flux reconstruction timing: %g seconds' % toc())

    # Compute error estimate using equilibrated stress reconstruction
    v = TestFunction(FunctionSpace(mesh, 'Discontinuous Lagrange', 0))
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
    info_blue('Error estimates: overall %g, discretization %g, regularization %g'
              % (Est_tot, Est_h, Est_eps))

    if exact_solution:
        S_exact = ufl.replace(S, {u: exact_solution})
        Est_up = sobolev_norm(S-S_exact, p1)
        info_blue('||R|| <= %g', Est_up)
    else:
        Est_up = None

    return u, Est_h, Est_eps, Est_tot, Est_up


def solve_p_laplace_adaptive_eps(p, criterion, V, f, df, zero_guess=False,
                                 exact_solution=None):
    """Approximate p-Laplace problem with rhs of the form

        (f, v) - (df, grad(v))  with test function v

    on space V with initial Newton approximation u0. Compute adaptively
    in regularization parameter eps and stop when

        lambda Est_h, Est_eps, Est_tot, Est_up: bool

    and return u."""
    eps = 1.0
    eps_decrease = 0.1**0.5
    u = None
    while True:
        result = solve_p_laplace(p, eps, V, f, df,
                                 u if not zero_guess else None,
                                 exact_solution)
        u, Est_h, Est_eps, Est_tot, Est_up = result
        # FIXME: Consider adding local criterion
        if criterion(Est_h, Est_eps, Est_tot, Est_up):
            break
        eps *= eps_decrease
    return u


def solve_problem(p, mesh, f, exact_solution=None, zero_guess=False):
    p1 = p/(p-1) # Dual Lebesgue exponent

    # Get Galerkin approximation of p-Laplace problem -\Delta_p u = f
    V = FunctionSpace(mesh, 'Lagrange', 1)
    criterion = lambda Est_h, Est_eps, Est_tot, Est_up: \
            Est_eps <= 0.001*Est_tot and Est_eps <= 0.001
    u = solve_p_laplace_adaptive_eps(p, criterion, V, f,
                                     zero(mesh.geometry().dim()), zero_guess,
                                     exact_solution)

    # p-Laplacian flux of u
    S = inner(grad(u), grad(u))**(0.5*Constant(p)-1.0) * grad(u)

    # Global lifting of W^{-1, p'} functional R = f + div(S)
    # Compute p-Laplace lifting on the patch on higher degree element
    # FIXME: Consider adding spatial adaptivity to compute lifting
    V_high = FunctionSpace(mesh, 'Lagrange', 4)
    criterion = lambda Est_h, Est_eps, Est_tot, Est_up: \
            Est_eps <= 0.001*Est_tot and Est_eps <= 0.001
    parameters['form_compiler']['quadrature_degree'] = 8
    r_glob = solve_p_laplace_adaptive_eps(p, criterion, V_high, f, S,
                                          zero_guess)
    parameters['form_compiler']['quadrature_degree'] = -1

    # Compute cell-wise norm of global lifting
    P0 = FunctionSpace(mesh, 'Discontinuous Lagrange', 0)
    dr_glob = project((grad(r_glob)**2)**Constant(0.5*p), P0)
    N = mesh.topology().dim() + 1 # vertices per cell
    dr_glob.vector().__imul__(N)
    plot(dr_glob)
    r_norm_glob = sobolev_norm(r_glob, p)**(p/p1)
    try:
        e_norm = assemble(dr_glob*dx)
        assert np.isclose(e_norm, N*r_norm_glob**p1)
    except AssertionError:
        info_red(r"||\nabla r||_p^p = %g, ||\nabla r||_p^p = %g"
                % (e_norm, N*r_norm_glob**p1))
    info_blue(r"||\nabla r|| = %g" % r_norm_glob)

    # Lower estimate on ||R|| using exact solution
    if exact_solution:
        v = TestFunction(V)
        R = ( inner(S, grad(v)) - f*v ) * dx(mesh)
        # FIXME: Possible cancellation due to underintegrating?!
        r_norm_glob_low = assemble(action(R, u)-action(R, exact_solution)) \
                / sobolev_norm(u - exact_solution, p)
        info_blue("||R|| >= %g" % r_norm_glob_low)

    # P1 lifting of local residuals
    P1 = V
    assert P1.ufl_element().family() == 'Lagrange' \
            and P1.ufl_element().degree() == 1
    r_loc_p1 = Function(P1)
    r_loc_p1_dofs = r_loc_p1.vector()
    v2d = vertex_to_dof_map(P1)

    # Alterative local lifting of residual
    # FIXME: Does this have a sense?
    r_norm_loc = 0.0
    #r_loc = Function(V)
    #r_loc_temp = Function(V)
    P4 = FunctionSpace(mesh, 'Lagrange', 4)
    r_loc = Function(P4)
    r_loc_temp = Function(P4)

    # Adjust verbosity
    old_log_level = get_log_level()
    set_log_level(WARNING)
    prg = Progress('Solving local liftings on patches', mesh.num_vertices())

    cf = CellFunction('size_t', mesh)
    for v in vertices(mesh):
        # Prepare submesh covering a patch
        cf.set_all(0)
        for c in cells(v):
            cf[c] = 1
        submesh = SubMesh(mesh, cf, 1)

        # Compute p-Laplace lifting on the patch on higher degree element
        # FIXME: Consider adding spatial adaptivity to compute lifting
        V_loc = FunctionSpace(submesh, 'Lagrange', 4)
        criterion = lambda Est_h, Est_eps, Est_tot, Est_up: \
                Est_eps <= 0.001*Est_tot and Est_eps <= 0.001
        parameters['form_compiler']['quadrature_degree'] = 8
        r = solve_p_laplace_adaptive_eps(p, criterion, V_loc, f, S, zero_guess)
        parameters['form_compiler']['quadrature_degree'] = -1

        # Compute local norm of residual
        r_norm_loc_a = sobolev_norm(r, p)**p
        r_norm_loc += r_norm_loc_a
        scale = (mesh.topology().dim() + 1) / sum(c.volume() for c in cells(v))
        r_loc_p1_dofs[v2d[v.index()]] = r_norm_loc_a * scale
        info_blue(r"||\nabla r_a|| = %g" % r_norm_loc_a**(1.0/p))

        # Alternative local lifting
        r = Extension(r, domain=mesh, element=r.ufl_element())
        r_loc_temp.interpolate(r)
        #project(r, V=r_loc_temp.function_space(), function=r_loc_temp, solver_type='lu')
        r_loc.vector()[:] += r_loc_temp.vector()
        #plot(r, title='r_a')
        #plot(r, mesh=submesh, title='r_a')
        #plot(r, mesh=mesh, title='r_a')
        plot(r_loc_temp, title='r_a')
        plot(r_loc, title='\sum_a r_a')
        #interactive()

        # Lower estimate on ||R||_a using exact solution
        if exact_solution:
            # FIXME: Treat hat as P1 test function to save resources

            # Prepare hat function on submesh
            parent_indices = submesh.data().array('parent_vertex_indices', 0)
            submesh_index = np.where(parent_indices == v.index())[0][0]
            assert parent_indices[submesh_index] == v.index()
            hat = hat_function(Vertex(submesh, submesh_index))

            # Compute residual lower bound
            phi = TestFunction(V_loc)
            R = lambda phi: ( inner(S, grad(phi)) - f*phi ) * dx(submesh)
            r_norm_loc_low = assemble(R(hat*u) - R(hat*exact_solution)) \
                    / sobolev_norm(hat*(u - exact_solution), p, submesh)
            info_blue("||R||_a >= %g" % r_norm_loc_low)

        # Advance progress bar
        set_log_level(PROGRESS)
        prg += 1
        set_log_level(WARNING)

    # Recover original verbosity
    set_log_level(old_log_level)

    r_norm_loc **= 1.0/p1
    plot(r_loc, title='Alternative local lifting')
    plot(r_loc_p1, title='P1 local lifting')
    try:
        e_norm = assemble(r_loc_p1*dx)
        assert np.isclose(e_norm, r_norm_loc**p1)
    except AssertionError:
        info_red(r"||e||_q^q = %g, \sum_a ||\nabla r_a||_p^p = %g"
                % (e_norm, r_norm_loc**p1))

    info_blue(r"||\nabla r||_p^(p-1) = %g, ( \sum_a ||\nabla r_a||_p^p )^(1/q) = %g"
              % (r_norm_glob, r_norm_loc))

    N = mesh.topology().dim() + 1 # vertices per cell
    C_PF = poincare_friedrichs_cutoff(mesh, p)
    ratio_a = r_norm_glob / ( N**(1.0/p) * C_PF * r_norm_loc )
    ratio_b = r_norm_loc / ( N**(1.0/p1) * r_norm_glob )
    info_blue("C_{cont,PF} = %g" %  C_PF)
    if ratio_a <= 1.0:
        info_green("(3.7a) ok: lhs/rhs = %g <= 1" % ratio_a)
    else:
        info_red("(3.7a) bad: lhs/rhs = %g > 1" % ratio_a)
    if ratio_b <= 1.0:
        info_green("(3.7b) ok: lhs/rhs = %g <= 1" % ratio_b)
    else:
        info_red("(3.7b) bad: lhs/rhs = %g > 1" % ratio_b)

    interactive()


def sobolev_norm(u, p, domain=None):
    p = Constant(p) if p is not 2 else p
    dX = dx(domain)
    return assemble(inner(grad(u), grad(u))**(p/2)*dX)**(1.0/float(p))


class Extension(Expression):
    def __init__(self, u, **kwargs):
        self._u = u
        assert 'domain' in kwargs and 'element' in kwargs
    def eval(self, values, x):
        try:
            self._u.eval(values, x)
        except RuntimeError:
            values[:] = 0.0
        else:
            assert not self._u.get_allow_extrapolation()


if __name__ == '__main__':
    from dolfintape.demo_problems.exact_solutions import pLaplace_modes, \
            pLaplace_ChaillouSuri, pLaplace_CarstensenKlose
    from dolfintape.mesh_fixup import mesh_fixup
    import mshr

    # UFLACS issue #49
    #parameters['form_compiler']['representation'] = 'uflacs'
    #parameters['linear_algebra_backend'] = 'Eigen'

    # -------------------------------------------------------------------------
    # Tests on unit square
    # -------------------------------------------------------------------------
    mesh = UnitSquareMesh(5, 5, 'crossed')
    #mesh = UnitSquareMesh(10, 10, 'crossed')
    #mesh = mesh_fixup(mesh)

    #u, f = pLaplace_modes(p=11.0, eps=0.0, m=1, n=1, domain=mesh, degree=4)
    #solve_problem(11.0, mesh, f, u)
    #u, f = pLaplace_modes(p=2, eps=0.0, m=1, n=1, domain=mesh, degree=4)
    #solve_problem(1.35, mesh, f)
    #u, f = pLaplace_modes(p=2, eps=0.0, m=1, n=1, domain=mesh, degree=4)
    #solve_problem(1.1, mesh, f, zero_guess=True)

    #p = 10.0
    #u, f = pLaplace_ChaillouSuri(p, domain=mesh, degree=4)
    #solve_problem(p, mesh, f, u)
    #p = 1.5
    #u, f = pLaplace_ChaillouSuri(p, domain=mesh, degree=4)
    #solve_problem(p, mesh, f, u)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Tests on L-shaped domain
    # -------------------------------------------------------------------------
    b0 = mshr.Rectangle(Point(0.0, 0.0), Point(0.5, 1.0))
    b1 = mshr.Rectangle(Point(0.0, 0.0), Point(1.0, 0.5))
    mesh = mshr.generate_mesh(b0 + b1, 10)
    mesh = mesh_fixup(mesh)

    f = Expression("1.+cos(2.*pi*x[0])*sin(2.*pi*x[1])", domain=mesh, degree=2)

    #solve_problem(11.0, mesh, f)
    #solve_problem(1.35, mesh, f)
    #solve_problem(1.1, mesh, f, zero_guess=True)

    b0 = mshr.Rectangle(Point(-1.0, -1.0), Point(1.0, 1.0))
    b1 = mshr.Rectangle(Point(0.0, -1.0), Point(1.0, 0.0))
    mesh = mshr.generate_mesh(b0 - b1, 10)
    #mesh = mshr.generate_mesh(b0 - b1, 40)
    mesh = mesh_fixup(mesh)
    p = 4.0
    u, f = pLaplace_CarstensenKlose(p=p, eps=0.0, delta=7.0/8, domain=mesh, degree=4)
    plot(u, mesh=mesh); plot(f, mesh=mesh)
    # There are some problems with quadrature element,
    # see https://bitbucket.org/fenics-project/ffc/issues/84,
    # so project to Lagrange
    f = project(f, FunctionSpace(mesh, 'Lagrange', 4))
    plot(f); interactive()
    solve_problem(p, mesh, f, u)
    # -------------------------------------------------------------------------

    list_timings(TimingClear_keep, [TimingType_wall])
