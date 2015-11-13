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
    bc = DirichletBC(V, 0.0, lambda x, onb: onb)
    solve(F_eps == 0, u, bc,
          solver_parameters={'newton_solver':
                                {'maximum_iterations': 500,
                                 'report': False}
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


def solve_problem(p, epsilons, mesh, f, exact_solution=None, zero_guess=False):
    p1 = p/(p-1) # Dual Lebesgue exponent
    V = FunctionSpace(mesh, 'Lagrange', 1)

    u = None
    for eps in epsilons:
        # TODO: No need for iteration if zero_guess
        u = solve_p_laplace(p, eps, V, f, zero(mesh.geometry().dim()),
                            None if zero_guess else u, exact_solution)[0]

    # p-Laplacian flux of u
    S = inner(grad(u), grad(u))**(0.5*p-1.0) * grad(u)

    # Global lifting of W^{-1, p'} functional R = f + div(S)
    V_high = FunctionSpace(mesh, 'Lagrange', 4)
    r_glob = None
    for eps in epsilons[:6]:
        parameters['form_compiler']['quadrature_degree'] = 8
        r_glob = solve_p_laplace(p, eps, V_high, f, S,
                                 None if zero_guess else r_glob)[0]
        parameters['form_compiler']['quadrature_degree'] = -1
        plot(r_glob)
        r_norm_glob = sobolev_norm(r_glob, p)**(p/p1)
        info_blue("||r|| = %g" % r_norm_glob)

    # Lower estimate on ||R|| using exact solution
    if exact_solution:
        v = TestFunction(V)
        R = ( inner(S, grad(v)) - f*v ) * dx(mesh)
        # FIXME: Possible cancellation due to underintegrating?!
        r_norm_glob_low = assemble(action(R, u)-action(R, exact_solution)) \
                / sobolev_norm(u - exact_solution, p)
        info_blue("||R|| >= %g" % r_norm_glob_low)

    #interactive()


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
    cf = CellFunction('size_t', mesh)
    #r_loc = Function(V)
    #r_loc_temp = Function(V)
    P4 = FunctionSpace(mesh, 'Lagrange', 4)
    r_loc = Function(P4)
    r_loc_temp = Function(P4)

    # Adjust verbosity
    old_log_level = get_log_level()
    set_log_level(WARNING)
    prg = Progress('Solving local liftings on patches', mesh.num_vertices())

    for v in vertices(mesh):
        cf.set_all(0)
        for c in cells(v):
            cf[c] = 1
        submesh = SubMesh(mesh, cf, 1)
        V_loc = FunctionSpace(submesh, 'Lagrange', 4)
        for eps in epsilons[:6]:
            #parameters['form_compiler']['quadrature_degree'] = 4
            parameters['form_compiler']['quadrature_degree'] = 8
            r = solve_p_laplace(p, eps, V_loc, f, S)[0]
            parameters['form_compiler']['quadrature_degree'] = -1
            #plot(r)

        # Compute local norm of residual
        r_norm_loc_a = sobolev_norm(r, p)**p
        r_norm_loc += r_norm_loc_a
        scale = (mesh.topology().dim() + 1) / sum(c.volume() for c in cells(v))
        r_loc_p1_dofs[v2d[v.index()]] = r_norm_loc_a * scale
        info_blue("||r_a|| = %g" % r_norm_loc_a)

        # Alternative local lifting
        r = Extension(r, domain=mesh, element=r.ufl_element())
        r_loc_temp.interpolate(r)
        #project(r, V=r_loc_temp.function_space(), function=r_loc_temp, solver_type='lu')
        r_loc.vector()[:] += r_loc_temp.vector()
        #plot(r, title='r_a')
        #plot(r, mesh=submesh, title='r_a')
        #plot(r, mesh=mesh, title='r_a')
        #plot(r_loc_temp, title='r_a')
        #plot(r_loc, title='\sum_a r_a')
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
    from dolfintape.demo_problems.exact_solutions import pLaplace_modes
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
    #solve_problem(11.0, [10.0**i for i in np.arange(1.0,  -6.0, -0.5)], mesh, f, u)
    #u, f = pLaplace_modes(p=1.35, eps=0.0, m=1, n=1, domain=mesh, degree=4)
    #solve_problem( 1.35, [10.0**i for i in np.arange(0.0, -22.0, -2.0)], mesh, f, u)
    #u, f = pLaplace_modes(p=1.1, eps=0.0, m=1, n=1, domain=mesh, degree=4)
    #solve_problem( 1.1,  [10.0**i for i in np.arange(0.0, -22.0, -2.0)], mesh, f, u, zero_guess=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Tests on L-shaped domain
    # -------------------------------------------------------------------------
    b0 = mshr.Rectangle(Point(0.0, 0.0), Point(0.5, 1.0))
    b1 = mshr.Rectangle(Point(0.0, 0.0), Point(1.0, 0.5))
    mesh = mshr.generate_mesh(b0 + b1, 10)
    mesh = mesh_fixup(mesh)

    f = Expression("1.+cos(2.*pi*x[0])*sin(2.*pi*x[1])", domain=mesh, degree=2)

    #solve_problem(11.0,  [10.0**i for i in np.arange(1.0,  -6.0, -0.5)], mesh, f)
    solve_problem( 1.35, [10.0**i for i in np.arange(0.0, -22.0, -2.0)], mesh, f)
    #solve_problem( 1.1,  [10.0**i for i in np.arange(0.0, -22.0, -2.0)], mesh, f, zero_guess=True)
    # -------------------------------------------------------------------------

    list_timings(TimingClear_keep, [TimingType_wall])
