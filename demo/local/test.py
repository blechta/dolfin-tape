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
import mshr
import ufl

from dolfintape import FluxReconstructor, CellDiameters
from dolfintape.poincare import poincare_friedrichs_cutoff
from dolfintape.mesh_fixup import mesh_fixup


# UFLACS issue #49
#parameters['form_compiler']['representation'] = 'uflacs'

# Prepare L-shaped mesh
b0 = mshr.Rectangle(Point(0.0, 0.0), Point(0.5, 1.0))
b1 = mshr.Rectangle(Point(0.0, 0.0), Point(1.0, 0.5))
mesh = mshr.generate_mesh(b0 + b1, 80)
mesh = mesh_fixup(mesh)

# Prepare common space and rhs
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
        rhs = ufl.replace(rhs, {rhs.arguments()[0]: TestFunction(V)})
        L, f = rhs, None

    # Problem formulation
    E = 1./p*(eps + dot(grad(u), grad(u)))**(0.5*p)*dx - action(L, u)
    F = derivative(E, u)
    bc = DirichletBC(V, 0.0, lambda x, onb: onb)
    solve(F == 0, u, bc,
          solver_parameters={'newton_solver':
                                {'maximum_iterations': 200,
                                 'report': False}
                            })

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

    u = None
    for eps in epsilons:
        u = solve_p_laplace(p, eps, V, f, None if zero_guess else u)[0]

    # Define residal form
    eps = Constant(0.0)
    E = ( 1./Constant(p)*(eps + dot(grad(u), grad(u)))**(0.5*Constant(p)) - f*u ) * dx
    R = derivative(E, u)

    # Global lifting of residual
    V_high = FunctionSpace(mesh, 'Lagrange', 4)
    r_glob = None
    for eps in epsilons[:6]:
        parameters['form_compiler']['quadrature_degree'] = 8
        r_glob = solve_p_laplace(p, eps, V_high, R)[0]
        parameters['form_compiler']['quadrature_degree'] = -1
        plot(r_glob)
        r_norm_glob = sobolev_norm(r_glob, p)**(p/p1)
        info("||r|| = %g" % r_norm_glob)
    #interactive()

    # Local lifting of residual
    r_norm_loc = 0.0
    cf = CellFunction('size_t', mesh)
    r_loc = Function(V)
    r_loc_temp = Function(V)

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
        eps = Constant(0.0)
        E = ( 1./Constant(p)*(eps + dot(grad(u), grad(u)))**(0.5*Constant(p)) - f*u ) * dx(submesh)
        R = derivative(E, u, TestFunction(V_loc))
        for eps in epsilons[:6]:
            #parameters['form_compiler']['quadrature_degree'] = 4
            parameters['form_compiler']['quadrature_degree'] = 8
            r = solve_p_laplace(p, eps, V_loc, R)[0]
            parameters['form_compiler']['quadrature_degree'] = -1
            #plot(r)
        r_norm_loc += sobolev_norm(r, p)**p
        # FIXME: Extrapolating instead of extending by zero?!
        r.set_allow_extrapolation(True)
        r_loc_temp.interpolate(r)
        r_loc_vec = r_loc.vector()
        r_loc_vec += r_loc_temp.vector()

        # Advance progress bar
        # TODO: This is possibly very slow!
        set_log_level(PROGRESS)
        prg += 1
        set_log_level(WARNING)

    # Recover original verbosity
    set_log_level(old_log_level)

    r_norm_loc **= 1.0/p1
    plot(r_loc)

    info(r"||\nabla r||_p^(p-1) = %g, ( \sum_a ||\nabla r_a||_p^p )^(1/q) = %g"
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

def sobolev_norm(u, p):
    p = Constant(p) if p is not 2 else p
    return assemble(inner(grad(u), grad(u))**(p/2)*dx)**(1.0/float(p))


if __name__ == '__main__':
    import numpy as np

    # p = 11.0
    solve_problem(11.0, [10.0**i for i in np.arange(1.0,  -6.0, -0.5)])

    # p = 1.1; works better with zero guess
    solve_problem( 1.1, [10.0**i for i in np.arange(0.0, -22.0, -2.0)], zero_guess=True)
