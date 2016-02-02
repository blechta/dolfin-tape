# Copyright (C) 2015, 2016 Jan Blechta
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

from __future__ import print_function

from dolfin import *
import ufl, ufc
import numpy as np
import os

from dolfintape import FluxReconstructor, CellDiameters
from dolfintape.poincare import poincare_friedrichs_cutoff
from dolfintape.hat_function import hat_function
from dolfintape.plotting import plot_alongside, pyplot
from dolfintape.utils import mkdir_p, logn, list_timings


not_working_in_parallel('This')

# UFLACS issue #49
#parameters['form_compiler']['representation'] = 'uflacs'

parameters['form_compiler']['optimize'] = True

parameters['plotting_backend'] = 'matplotlib'


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


class FunctionSpaceCache(dict):
    def __setitem__(self, key, value):
        mesh, family, degree = key
        dict.__setitem__(self, (mesh.id(), family, degree), value)
    def __getitem__(self, key):
        mesh, family, degree = key
        try:
            return dict.__getitem__(self, (mesh.id(), family, degree))
        except KeyError:
            self[key] = space = FunctionSpace(mesh, family, degree)
            return space
function_space_cache = FunctionSpaceCache()


boundary = CompiledSubDomain("on_boundary")


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
    u = u0.copy(deepcopy=False) if u0 else Function(V)

    # Problem formulation
    S = inner(grad(u), grad(u))**(0.5*p-1.0) * grad(u) + df
    S_eps = (eps + inner(grad(u), grad(u)))**(0.5*p-1.0) * grad(u) + df
    v = TestFunction(V)
    F_eps = ( inner(S_eps, grad(v)) - f*v ) * dx
    bc = DirichletBC(V, exact_solution if exact_solution else 0.0, boundary)
    solve(F_eps == 0, u, bc,
          solver_parameters={'newton_solver':
                                {'maximum_iterations': 50,
                                 'report': False}
                            })

    # Reconstruct flux q in H^p1(div) s.t.
    #       q ~ -S
    #   div q ~ f
    reconstructor = reconstructor_cache[(V.mesh(), V.ufl_element().degree())]
    q = reconstructor.reconstruct(S, f).sub(0, deepcopy=False)

    # Compute error estimate using equilibrated stress reconstruction
    P0 = function_space_cache[(mesh, 'Discontinuous Lagrange', 0)]
    v = TestFunction(P0)
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

    # Wrap arrays as cell functions
    est_h = vecarray_to_cellfunction(est_h, P0)
    est_eps = vecarray_to_cellfunction(est_eps, P0)
    est_tot = vecarray_to_cellfunction(est_tot, P0)

    # Upper estimate using exact solution
    if exact_solution:
        S_exact = ufl.replace(S, {u: exact_solution})
        Est_up = sobolev_norm(S-S_exact, p1, k=0)
    else:
        Est_up = None

    log(18, 'Error estimates: overall %g, discretization %g, '
            'regularization %g, estimate_up %s'
            % (Est_tot, Est_h, Est_eps, Est_up))

    return u, est_h, est_eps, est_tot, Est_h, Est_eps, Est_tot, Est_up


def solve_p_laplace_adaptive_eps(p, criterion, u, f, df, zero_guess=False,
                                 exact_solution=None):
    """Approximate p-Laplace problem with rhs of the form

        (f, v) - (df, grad(v))  with test function v

    on space V with initial Newton approximation u0. Compute adaptively
    in regularization parameter eps and stop when

        criterion = lambda u_h, Est_h, Est_eps, Est_tot, Est_up: bool

    returns True. Return u and various estimators."""
    # Initial parameters
    # FIXME: Shouldn't we retain the state of eps from the previous
    #        spatial adaptivity step?
    #eps, eps_decrease = 1.0, 0.1**0.5
    eps, eps_decrease = 1e-3, 0.1**1.5
    #eps, eps_decrease = 1e-6, 0.1**0.5

    logn(25, 'Adapting regularization')
    while True:
        logn(25, '.')
        result = solve_p_laplace(p, eps, u.function_space(), f, df,
                                 u if not zero_guess else None,
                                 exact_solution)
        u, est_h, est_eps, est_tot, Est_h, Est_eps, Est_tot, Est_up = result
        if criterion(None, Est_h, Est_eps, Est_tot, Est_up):
            break
        eps *= eps_decrease
    log(25, '')
    return result


def solve_p_laplace_adaptive_mesh(p, criterion, V, f, df, zero_guess=False,
                                  exact_solution=None):
    """Approximate p-Laplace problem with rhs of the form

        (f, v) - (df, grad(v))  with test function v

    with initial Newton approximation u0. Start on space V and refine
    adaptively mesh (where discretization estimator is large) and
    regularization parameter epsilon until

        criterion = lambda u_h, Est_h, Est_eps, Est_tot, Est_up: bool

    returns True. Return u."""
    u = Function(V)
    while True:
        logn(25, 'Adapting mesh (space dimension %s): ' % V.dim())
        result = solve_p_laplace_adaptive_eps(p, criterion, u, f, df,
                                 zero_guess, exact_solution)
        u, est_h, est_eps, est_tot, Est_h, Est_eps, Est_tot, Est_up = result

        # HACK: Drop unneeded data
        reconstructor_cache.clear()
        function_space_cache.clear()

        # Check convergence
        log(25, 'Estimators h, eps, tot, up: %s' % (result[4:],))
        log(25, r'||\nabla u_h||_p^{p-1} = %s' % sobolev_norm(u, p)**(p-1.0))
        if criterion(u, Est_h, Est_eps, Est_tot, Est_up):
            break

        # Refine mesh
        markers = estimator_to_markers(est_h, p/(p-1.0), aggressivity=2.5)
        log(25, 'Marked %s of %s cells for refinement'
                % (sum(markers), markers.mesh().num_cells()))
        mesh = adapt(V.mesh(), markers)
        u = adapt(u, mesh)
        V = u.function_space()

    return u


def solve_problem(p, mesh, f, exact_solution=None, zero_guess=False):
    p1 = p/(p-1) # Dual Lebesgue exponent

    # Get Galerkin approximation of p-Laplace problem -\Delta_p u = f
    V = FunctionSpace(mesh, 'Lagrange', 1)
    criterion = lambda u_h, Est_h, Est_eps, Est_tot, Est_up: Est_eps <= 1e-6*Est_tot
    log(25, 'Computing residual of p-Laplace problem')
    u = solve_p_laplace_adaptive_mesh(p, criterion, V, f,
                                     zero(mesh.geometry().dim()), zero_guess,
                                     exact_solution)

    # p-Laplacian flux of u
    S = inner(grad(u), grad(u))**(0.5*Constant(p)-1.0) * grad(u)

    # Global lifting of W^{-1, p'} functional R = f + div(S)
    # Compute p-Laplace lifting on the patch on higher degree element
    V_high = FunctionSpace(mesh, 'Lagrange', 2)
    criterion = lambda u_h, Est_h, Est_eps, Est_tot, Est_up: \
        Est_eps <= 1e-2*Est_tot and Est_tot <= 1e-3*sobolev_norm(u_h, p)**(p-1.0)
    parameters['form_compiler']['quadrature_degree'] = 8
    log(25, 'Computing global lifting of the resiual')
    u.set_allow_extrapolation(True)
    r_glob = solve_p_laplace_adaptive_mesh(p, criterion, V_high, f, S,
                                          zero_guess)
    u.set_allow_extrapolation(False)
    parameters['form_compiler']['quadrature_degree'] = -1

    # Compute cell-wise norm of global lifting
    dr_glob_fine, dr_glob_coarse = compute_cellwise_grad(r_glob, p)
    r_norm_glob = sobolev_norm(r_glob, p)**(p/p1)
    try:
        e_norm_coarse = assemble(dr_glob_coarse*dx)
        e_norm_fine = assemble(dr_glob_fine*dx)
        N = mesh.topology().dim() + 1 # vertices per cell
        assert np.isclose(e_norm_fine, N*r_norm_glob**p1)
        assert np.isclose(e_norm_coarse, N*r_norm_glob**p1)
    except AssertionError:
        info_red(r"N*||\nabla r||_p^p = %g, %g, %g"
                % (e_norm_coarse, e_norm_fine, N*r_norm_glob**p1))
    else:
        info_blue(r"||\nabla r||_p^{p-1} = %g" % r_norm_glob)

    # Compute patch-wise norm of global lifting
    P1 = V
    assert P1.ufl_element().family() == 'Lagrange' \
            and P1.ufl_element().degree() == 1
    dr_glob_p1 = Function(P1)
    dr_glob_p1_vec = dr_glob_p1.vector()
    x = np.ndarray(mesh.geometry().dim())
    val = np.ndarray(1)
    c_ufc = ufc.cell() # FIXME: Is empty ufc cell sufficient?
    v2d = vertex_to_dof_map(P1)
    for c in cells(mesh):
        dr_glob_coarse.eval(val, x, c, c_ufc)
        for v in vertices(c):
            # FIXME: it would be better to assemble vector once
            vol_cell = c.volume()
            vol_patch = sum(c.volume() for c in cells(v))
            dof = v2d[v.index()]
            dr_glob_p1_vec[dof] = dr_glob_p1_vec[dof][0] \
                                + val[0]*vol_cell/vol_patch

    # Lower estimate on ||R|| using exact solution
    if exact_solution:
        v = TestFunction(V)
        R = ( inner(S, grad(v)) - f*v ) * dx(mesh)
        # FIXME: Possible cancellation due to underintegrating?!
        r_norm_glob_low = assemble(action(R, u)-action(R, exact_solution)) \
                / sobolev_norm(u - exact_solution, p)
        info_blue("||R||_{-1,q} >= %g (estimate using exact solution)"
                  % r_norm_glob_low)

    # P1 lifting of local residuals
    P1 = V
    assert P1.ufl_element().family() == 'Lagrange' \
            and P1.ufl_element().degree() == 1
    r_loc_p1 = Function(P1)
    r_loc_p1_dofs = r_loc_p1.vector()
    v2d = vertex_to_dof_map(P1)

    # Alterative local lifting of residual
    # FIXME: Does this have a sense? Do we need it?
    r_norm_loc = 0.0
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
        V_loc = FunctionSpace(submesh, 'Lagrange', 4)
        criterion = lambda u_h, Est_h, Est_eps, Est_tot, Est_up: \
            Est_eps <= 1e-2*Est_tot and Est_tot <= 1e-3*sobolev_norm(u_h, p)**(p-1.0)
        parameters['form_compiler']['quadrature_degree'] = 8
        r = solve_p_laplace_adaptive_mesh(p, criterion, V_loc, f, S, zero_guess)
        parameters['form_compiler']['quadrature_degree'] = -1

        # Compute local norm of residual
        r_norm_loc_a = sobolev_norm(r, p)**p
        r_norm_loc += r_norm_loc_a
        scale = (mesh.topology().dim() + 1) / sum(c.volume() for c in cells(v))
        r_loc_p1_dofs[v2d[v.index()]] = r_norm_loc_a * scale
        log(18, r"||\nabla r_a||_p = %g" % r_norm_loc_a**(1.0/p))

        # Alternative local lifting
        r = Extension(r, domain=mesh, element=r.ufl_element())
        r_loc_temp.interpolate(r)
        #project(r, V=r_loc_temp.function_space(), function=r_loc_temp, solver_type='lu')
        r_loc.vector()[:] += r_loc_temp.vector()

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
                    / sobolev_norm(hat*(u - exact_solution), p, domain=submesh)
            log(18, r"||R||_{-1,q,\omega_a} >= %g (estimate using exact solution)"
                    % r_norm_loc_low)

        # Advance progress bar
        set_log_level(PROGRESS)
        prg += 1
        set_log_level(WARNING)

    # Recover original verbosity
    set_log_level(old_log_level)

    r_norm_loc **= 1.0/p1
    try:
        e_norm = assemble(r_loc_p1*dx)
        assert np.isclose(e_norm, r_norm_loc**p1)
    except AssertionError:
        info_red(r"||e||_q^q = %g, \sum_a ||\nabla r_a||_p^p = %g"
                % (e_norm, r_norm_loc**p1))

    info_blue(r"||\nabla r||_p^{p-1} = %g, ( \sum_a ||\nabla r_a||_p^p )^{1/q} = %g"
              % (r_norm_glob, r_norm_loc))

    N = mesh.topology().dim() + 1 # vertices per cell
    C_PF = poincare_friedrichs_cutoff(mesh, p)
    ratio_a = ( N**(1.0/p) * C_PF * r_norm_loc ) / r_norm_glob
    ratio_b = ( N**(1.0/p1) * r_norm_glob ) / r_norm_loc
    info_blue("C_{cont,PF} = %g" %  C_PF)
    if ratio_a >= 1.0:
        info_green("(3.7a) ok: rhs/lhs = %g >= 1" % ratio_a)
    else:
        info_red("(3.7a) bad: rhs/lhs = %g < 1" % ratio_a)
    if ratio_b >= 1.0:
        info_green("(3.7b) ok: rhs/lhs = %g >= 1" % ratio_b)
    else:
        info_red("(3.7b) bad: rhs/lhs = %g < 1" % ratio_b)

    return dr_glob_p1, r_loc_p1


def sobolev_norm(u, p, k=1, domain=None):
    if u is None:
        return np.infty
    p = Constant(p) if p is not 2 else p
    dX = dx(domain)
    if k == 1:
        u = grad(u)
    elif k != 0:
        raise NotImplementedError
    return assemble(inner(u, u)**(p/2)*dX)**(1.0/float(p))


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


def vecarray_to_cellfunction(arr, space, cf=None):
    assert space.ufl_element().family() == "Discontinuous Lagrange"
    assert space.ufl_element().degree() == 0
    assert space.ufl_element().value_shape() == ()

    if cf is None:
        cf = CellFunction('double', space.mesh())
    else:
        assert isinstance(cf, CellFunctionDouble)

    cell_dofs = space.dofmap().cell_dofs
    for c in cells(space.mesh()):
        cf[c] = arr[cell_dofs(c.index())[0]]

    return cf


def estimator_to_markers(est, p1, cf=None, aggressivity=1.0):
    assert isinstance(est, CellFunctionDouble)

    if cf is None:
        cf = CellFunction('bool', est.mesh())
    else:
        assert isinstance(cf, CellFunctionBool)

    norm = sum(abs(v)**p1 for v in est)
    norm = MPI.sum(est.mesh().mpi_comm(), norm)
    norm = (norm/est.mesh().num_cells())**(1.0/p1)

    for c in cells(est.mesh()):
        cf[c] = abs(est[c]) > aggressivity*norm

    if sum(cf) == 0:
        estimator_to_markers(est, p1, cf=cf, aggressivity=0.5*aggressivity)

    return cf


def compute_cellwise_grad(r, p):
    # Compute desired quantity accurately on fine mesh
    mesh_fine = r.function_space().mesh()
    P0_fine = FunctionSpace(mesh_fine, 'Discontinuous Lagrange', 0)
    dr_fine = project((grad(r)**2)**Constant(0.5*p), P0_fine)

    # Some scaling
    N = mesh_fine.topology().dim() + 1 # vertices per cell
    dr_fine.vector().__imul__(N)

    # Compute parent cells from finest to coarsest
    mesh = mesh_fine
    tdim = mesh.topology().dim()
    parent_cells = slice(None)
    while mesh:
        parent_cells = mesh.data().array('parent_cell', tdim)[parent_cells]
        mesh = mesh.parent()

    # Init coarse quantity
    mesh_coarse = mesh_fine.root_node()
    P0_coarse = FunctionSpace(mesh_coarse, 'Discontinuous Lagrange', 0)
    dr_coarse = Function(P0_coarse)

    # Fetch needed objects to speed-up the hot loop
    dofs_fine = P0_fine.dofmap().cell_dofs
    dofs_coarse = P0_coarse.dofmap().cell_dofs
    x_fine = dr_fine.vector()
    x_coarse = dr_coarse.vector()

    # Reduce fine to coarse
    for c in cells(mesh_fine):
        i = c.index()
        scale = c.volume()/Cell(mesh_coarse, parent_cells[i]).volume()
        x_coarse[dofs_coarse(parent_cells[i])] += scale * x_fine[dofs_fine(i)]

    return dr_fine, dr_coarse


def plot_liftings(glob, loc, prefix):
    path = "results"
    mkdir_p(path)
    plot_alongside(glob, loc, mode="color", shading="flat", edgecolors="k")
    pyplot.savefig(os.path.join(path, prefix+"f.pdf"))
    plot_alongside(glob, loc, mode="color", shading="gouraud")
    pyplot.savefig(os.path.join(path, prefix+"g.pdf"))
    plot_alongside(glob, loc, mode="warp")
    pyplot.savefig(os.path.join(path, prefix+"w.pdf"))


def test_ChaillouSuri(p, N):
    from dolfintape.demo_problems.exact_solutions import pLaplace_ChaillouSuri

    mesh = UnitSquareMesh(N, N, 'crossed')
    u, f = pLaplace_ChaillouSuri(p, domain=mesh, degree=4)
    glob, loc = solve_problem(p, mesh, f, u)

    plot_liftings(glob, loc, 'ChaillouSuri_%s_%s' % (p, N))
    list_timings(TimingClear_clear, [TimingType_wall])


def test_CarstensenKlose(p, N):
    from dolfintape.demo_problems.exact_solutions import pLaplace_CarstensenKlose
    from dolfintape.mesh_fixup import mesh_fixup
    import mshr

    # Build mesh on L-shaped domain (-1, 1)^2 \ (0, 1)*(-1, 0)
    b0 = mshr.Rectangle(Point(-1.0, -1.0), Point(1.0, 1.0))
    b1 = mshr.Rectangle(Point(0.0, -1.0), Point(1.0, 0.0))
    mesh = mshr.generate_mesh(b0 - b1, N)
    mesh = mesh_fixup(mesh)

    u, f = pLaplace_CarstensenKlose(p=p, eps=0.0, delta=7.0/8,
                                    domain=mesh, degree=4)
    # There are some problems with quadrature element,
    # see https://bitbucket.org/fenics-project/ffc/issues/84,
    # so project to Lagrange element
    f = project(f, FunctionSpace(mesh, 'Lagrange', 4))
    f.set_allow_extrapolation(True)

    glob, loc = solve_problem(p, mesh, f, u)

    plot_liftings(glob, loc, 'CarstensenKlose_%s_%s' % (p, N))
    list_timings(TimingClear_clear, [TimingType_wall])


def main(argv):
    default_tests = [
            ('ChaillouSuri',   10.0,  5),
            ('ChaillouSuri',   10.0, 10),
            ('ChaillouSuri',   10.0, 15),
            ('ChaillouSuri',    1.5,  5),
            ('ChaillouSuri',    1.5, 10),
            ('ChaillouSuri',    1.5, 15),
            ('CarstensenKlose', 4.0,  5),
            ('CarstensenKlose', 4.0, 10),
            ('CarstensenKlose', 4.0, 15),
        ]

    usage = """%s

usage: python %s [-h|--help] [test-name p N]

Without arguments run default test cases. Or run test case with
given value of p when given on command-line.

Default test cases:

%s
""" % (__doc__, __file__, default_tests)

    # Decrease verbosity of DOLFIN
    set_log_level(25)

    # Run all tests
    if len(argv) == 1:
        for test in default_tests:
            exec('test_%s(%s)' % test)
            return

    # Print help
    if argv[1] in ['-h', '--help']:
        print(usage)
        return

    # Now expecting 3 arguments
    if len(argv) != 4:
        print("Command-line arguments not understood!")
        print()
        print(usage)
        return 1

    # Run the selected test
    try:
        exec("tester = test_%s" % argv[1])
    except NameError:
        print ("'Test %s' does not exist!" % argv[1])
        return 1
    else:
        tester(float(argv[2]), int(argv[3]))
        return


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
