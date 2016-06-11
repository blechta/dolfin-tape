# Copyright (C) 2016 Jan Blechta
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

from dolfin import *
import ufl
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain

from dolfintape import FluxReconstructor
from dolfintape.utils import logn
from dolfintape.cell_diameter import CellDiameters
from dolfintape.poincare import poincare_const
from dolfintape.sobolev_norm import sobolev_norm

__all__ = ['solve_p_laplace_adaptive', 'pLapLaplaceAdaptiveSolver',
           'geometric_progression']


def solve_p_laplace_adaptive(p, criterion, V, f, df, u_ex=None,
                             eps0=1.0, eps_decrease=0.1**0.5,
                             solver_parameters=None):
    """Approximate p-Laplace problem with rhs of the form

        (f, v) - (df, grad(v))  with test function v

    on initial space V. Compute adaptively in regularization parameter
    and refine mesh adaptively until

        criterion = lambda u_h, Est_h, Est_eps, Est_tot, Est_up: bool

    returns True. Return u."""
    if isinstance(p, Constant):
        q = Constant(float(p)/(float(p)-1.0))
    else:
        q = p/(p-1)
    eps = geometric_progression(eps0, eps_decrease)
    solver = pLaplaceAdaptiveSolver(p, q, f, df, u_ex)
    return solver.solve(V, criterion, eps, solver_parameters)


def geometric_progression(a0, q):
    """Return function returning generator taking values
    of geometric progression:
        a0,
        a0*q,
        a0*q**2,
        a0*q**3,
        ...
    """
    def generator():
        _a0 = a0
        while True:
            yield _a0
            _a0 *= q
    return generator


class pLaplaceAdaptiveSolver(object):
    """Adaptive solver for p-Laplace problem with spatial adaptivity and
    adaptivity in regularization parameter.
    """
    def __init__(self, p, q, f, df, exact_solution=None):
        """Adaptive solver for p-Laplace problem (q should be dual exponent
        q = p/(p-1)) with rhs of the form

            (f, v) - (df, grad(v))  with test function v.
        """
        assert np.allclose(1.0/float(p) + 1.0/float(q), 1.0), \
                "Expected conjugate Lebesgue exponents " \
                "p, q (of type int, float, or Constatnt)!"

        self.p = p
        self.q = q
        self.f = f
        self.df = df
        self.u_ex = exact_solution

        self.boundary = CompiledSubDomain("on_boundary")


    def solve(self, V, criterion, eps, solver_parameters=None):
        """Start on initial function space V, refine adaptively mesh and
        regularization parameter provided by decreasing generator eps
        until

            criterion = lambda u_h, Est_h, Est_eps, Est_tot, Est_up: bool

        returns True. Return found approximation."""
        p = float(self.p)
        u = Function(V)

        while True:
            logn(25, 'Adapting mesh (space dimension %s): ' % V.dim())
            result = self._adapt_eps(criterion, u, eps, solver_parameters)
            u, est_h, est_eps, est_tot, Est_h, Est_eps, Est_tot, Est_up = result

            # Check convergence
            log(25, 'Estimators h, eps, tot, up: %s' % (result[4:],))
            log(25, r'||\nabla u_h||_p^{p-1} = %s' % sobolev_norm(u, p)**(p-1.0))
            if criterion(u, Est_h, Est_eps, Est_tot, Est_up):
                break

            # Refine mesh
            markers = self.estimator_to_markers(est_h, p/(p-1.0), fraction=0.5)
            log(25, 'Marked %s of %s cells for refinement'
                    % (sum(markers), markers.mesh().num_cells()))
            adapt(V.mesh(), markers)
            mesh = V.mesh().child()
            adapt(u, mesh)
            u = u.child()
            V = u.function_space()

        return u


    def _solve(self, eps, u, reconstructor, P0, solver_parameters):
        """Find approximate solution with fixed eps and mesh. Use
        reconstructor to reconstruct the flux and estimate errors.
        """
        p, q = self.p, self.q
        f, df = self.f, self.df
        boundary = self.boundary
        exact_solution = self.u_ex
        V = u.function_space()
        mesh = V.mesh()
        dx = Measure('dx', domain=mesh)
        eps = Constant(eps)

        # Problem formulation
        S = inner(grad(u), grad(u))**(p/2-1) * grad(u) + df
        S_eps = (eps + inner(grad(u), grad(u)))**(p/2-1) * grad(u) + df
        v = TestFunction(V)
        F_eps = ( inner(S_eps, grad(v)) - f*v ) * dx
        bc = DirichletBC(V, exact_solution if exact_solution else 0.0, boundary)

        # Solve
        solve(F_eps == 0, u, bc, solver_parameters=solver_parameters or {})

        # Reconstruct flux q in H^q(div) s.t.
        #       q ~ -S
        #   div q ~ f
        Q = reconstructor.reconstruct(S, f).sub(0, deepcopy=False)

        # Compute error estimate using equilibrated stress reconstruction
        v = TestFunction(P0)
        h = CellDiameters(mesh)
        Cp = Constant(poincare_const(mesh.type(), p))
        est0 = assemble(((Cp*h*(f-div(Q)))**2)**(0.5*q)*v*dx)
        est1 = assemble(inner(S_eps+Q, S_eps+Q)**(0.5*q)*v*dx)
        est2 = assemble(inner(S_eps-S, S_eps-S)**(0.5*q)*v*dx)
        q = float(q)
        est_h   = est0.array()**(1.0/q) + est1.array()**(1.0/q)
        est_eps = est2.array()**(1.0/q)
        est_tot = est_h + est_eps
        Est_h   = MPI.sum( mesh.mpi_comm(), (est_h  **q).sum() )**(1.0/q)
        Est_eps = MPI.sum( mesh.mpi_comm(), (est_eps**q).sum() )**(1.0/q)
        Est_tot = MPI.sum( mesh.mpi_comm(), (est_tot**q).sum() )**(1.0/q)

        # Wrap arrays as cell functions
        est_h = self.vecarray_to_cellfunction(est_h, P0)
        est_eps = self.vecarray_to_cellfunction(est_eps, P0)
        est_tot = self.vecarray_to_cellfunction(est_tot, P0)

        # Upper estimate using exact solution
        if exact_solution:
            S_exact = ufl.replace(S, {u: exact_solution})
            Est_up = sobolev_norm(S-S_exact, q, k=0)
        else:
            Est_up = None

        log(18, 'Error estimates: overall %g, discretization %g, '
                'regularization %g, estimate_up %s'
                % (Est_tot, Est_h, Est_eps, Est_up))

        return u, est_h, est_eps, est_tot, Est_h, Est_eps, Est_tot, Est_up


    def _adapt_eps(self, criterion, u, epsilons, solver_parameters):
        """Solve adaptively in eps on fixed space (given by u) until

            criterion = lambda None, Est_h, Est_eps, Est_tot, Est_up: bool

        return True. Notice None supplied instead of u_h, thus not taking
        discretization error criterion into account.
        """
        # Prepare flux reconstructor and P0 space
        log(25, 'Initializing flux reconstructor')
        reconstructor = FluxReconstructor(u.function_space().mesh(),
                u.function_space().ufl_element().degree())
        P0 = FunctionSpace(u.function_space().mesh(),
                'Discontinuous Lagrange', 0)

        # Adapt regularization
        logn(25, 'Adapting regularization')
        for eps in epsilons():
            debug('Regularizing using eps = %s' % eps)
            logn(25, '.')
            result = self._solve(eps, u, reconstructor, P0, solver_parameters)
            u, est_h, est_eps, est_tot, Est_h, Est_eps, Est_tot, Est_up = result
            if criterion(None, Est_h, Est_eps, Est_tot, Est_up):
                break
        log(25, '')
        return result


    @staticmethod
    def estimator_to_markers(est, q, cf=None, fraction=0.5):
        """Take double CellFunction and convert it to bool cell function
        using Dorfler marking strategy.
        """
        assert isinstance(est, CellFunctionDouble)

        if cf is None:
            cf = CellFunction('bool', est.mesh())
        else:
            assert isinstance(cf, CellFunctionBool)

        # Take appropriate powers (operating on a copy)
        _est = MeshFunction('double', est)
        np.abs(_est.array(), out=_est.array())
        _est.array()[:] **= q

        # Call Dorfler marking
        not_working_in_parallel("Dorfler marking strategy")
        dorfler_mark(cf, _est, fraction)

        return cf


    @staticmethod
    def vecarray_to_cellfunction(arr, space, cf=None):
        """Convert numpy array coming from function.vector().array() for
        P0 function to CellFunction, optionally existing cf.
        """
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
