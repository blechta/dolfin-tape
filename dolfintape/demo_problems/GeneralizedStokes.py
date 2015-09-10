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

from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

from dolfintape import FluxReconstructor
from dolfintape.deviatoric_space import TensorFunctionSpace, deviatoric
from dolfintape.cell_diameter import CellDiameters
from dolfintape.poincare import poincare_const
from dolfintape.utils import adapt


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
        I_RTN = lambda x: x # Dummy interpolator
        return ( Constant(1.0/mu)*I_RTN(hat[p]*inner(stress_component, v))
                 -hat[p]*force_component*phi
                 +inner(grad(hat[p]), stress_component)*phi
                 +residual_component*phi )*self.dp(p)

    def a(self, p):
        """lhs of (6.7); one component"""
        d, q = TrialFunctions(self._W)
        v, phi = TestFunctions(self._W)
        mu, r = float(self._mu), float(self._r)
        return ( Constant(1.0/mu)*inner(d, v)
               + inner(q, div(v))
               + inner(phi, div(d)) )*self.dp(p)


class GeneralizedStokesProblem(object):

    def __init__(self, mesh, constitutive_law, f, eps):

        V = VectorFunctionSpace(mesh, 'CG', 2)
        Q = FunctionSpace(mesh, 'CG', 1)
        S = TensorFunctionSpace(mesh, 'DG', 1, symmetry=True, zero_trace=True)
        W = MixedFunctionSpace([V, Q, S])
        info_blue('Number DOFs: %d' % W.dim())

        # Interpolate possibly stored old solution to the new space
        try:
            w = self._w
        except AttributeError:
            w = Function(W)
        else:
            # FIXME: use LagrangeInterpolater for parallel functionality
            w.set_allow_extrapolation(True)
            w = interpolate(w, W)

        (u, p, s) = split(w)
        (v, q, t) = TestFunctions(W)
        s = deviatoric(s)
        t = deviatoric(t)

        d = sym(grad(u))
        g = constitutive_law.g()

        F = lambda eps: ( inner(s, grad(v)) - p*div(v) - q*div(u)
                        + inner(g(s, d, eps), t) - inner(f, v) )*dx
        bcs = self.bcs(W)

        self._mesh = mesh
        self._F = F
        self._bcs = bcs
        self._w = w
        self._constitutive_law = constitutive_law
        self._f = f
        self._eps = eps


    def refine(self, *args):
        # Refine mesh and facet function if any
        parameters["refinement_algorithm"] = "plaza_with_parent_facets"
        self._mesh = refine(self._mesh, *args)
        try:
            self._ff = adapt(self._ff, self._mesh)
        except AttributeError:
            pass
        # Forms and function adapted by constructor
        GeneralizedStokesProblem.__init__(self, self._mesh,
                                          self._constitutive_law,
                                          self.f, self._eps)


    def solve(self):
        F = self._F(self._eps)
        solve(F==0, self._w, bcs=self._bcs, J=derivative(F, self._w))
        return self._w


    def solve_adaptive_eps(self):
        while True:
            w = self.solve()
            if self.criterion_eps():
                info_blue('Regularization loop converged with eps = %g' % self._eps)
                break
        return w


    def solve_adaptive_h(self):
        while True:
            w = self.solve_adaptive_eps()
            if self.criterion_h():
                info_blue('Mesh refinement loop converged with %d DOFs'
                            % w.function_space().dim())
                break
        return w


    def criterion_eps(self):
        Eta_disc, Eta_reg, Eta_osc, eta_disc, eta_reg, eta_osc = \
                self.estimate_errors_components()
        if Eta_reg <= self.criterion_eps_threshold()*Eta_disc:
            return True
        else:
            # FIXME: Get rid of hardcoded value
            self._eps *= 0.1**0.5
            info_red('Decreasing eps to: %g' % self._eps)
            return False


    def criterion_h(self):
        Eta_1, Eta_2, Eta_3, eta_1, eta_2, eta_3 = \
                self.estimate_errors_overall()
        # FIXME: Get rid of hardcoded value
        #if Eta_1 < 1e-3 and Eta_2 < 1e-3 and Eta_3 < 1e-3:
        if Eta_1 < 0.1:
            return True
        else:
            cf = self.compute_refinement_markers(eta_1, eta_2, eta_3)
            self.refine(cf)
            return False


    def compute_refinement_markers(self, eta_1, eta_2, eta_3):
        # FIXME: This primitive algorithm is basically wrong
        # FIXME: Get rid of hardcoded value
        threshold = 0.95

        cf = CellFunction('bool', self._mesh)

        ufc_cell = dolfin.cpp.mesh.ufc.cell()
        x = np.array(self._mesh.geometry().dim()*(0.0,), dtype='float_')
        y = np.array((0.0,), dtype='float_')

        for eta in (eta_1, eta_2, eta_3):
            eta_max = eta.vector().max()
            for c in cells(self._mesh):
                c.get_cell_data(ufc_cell)
                eta.eval(y, x, c, ufc_cell)
                if y[0] > threshold * eta_max:
                    cf[c] = True

        #plot(cf, interactive=True)
        return cf


    def criterion_eps_threshold(self):
        # FIXME: Get rid of hardcoded value
        return 0.5


    def reconstructor(self):
        try:
            reconstructor = self._reconstructor
            assert reconstructor._mesh.id() == self._mesh.id()
        except (AttributeError, AssertionError):
            mesh = self._w.function_space().mesh()
            l = self._w.function_space().ufl_element().sub_elements()[0].degree()
            r = self._constitutive_law.r()
            mu = self._constitutive_law.mu()
            reconstructor = ImplicitStokesFluxReconstructor(mesh, l, r, mu)
            self._reconstructor = reconstructor
        return reconstructor


    def estimate_errors_overall(self):
        w = self._w
        mesh = self._w.function_space().mesh()
        comm = mesh.mpi_comm()
        reconstructor = self.reconstructor()
        f = self._f
        r = self._constitutive_law.r()

        u, p, s = w.split()
        s = deviatoric(s)
        d = sym(grad(u))

        g = self._constitutive_law.g()

        gdim = mesh.geometry().dim()

        # TODO: Consolidate notation
        q = []
        I = Identity(gdim)
        stress = -p*I + s
        algebraic_residual = 0.0
        for i in xrange(gdim):
            w = reconstructor.reconstruct(stress[i, :], f[i], algebraic_residual)
            q.append(w.sub(0, deepcopy=True))
        q = as_vector(q)

        # Poincare constant on cells
        cell_type = mesh.type()
        C_P = poincare_const(cell_type, r, gdim)
        h = CellDiameters(mesh)

        # Dual Lebesgue exponent; denoted s in the paper
        r1 = r/(r-1)
        # r/2 is dangerous for integer r != 2
        assert isinstance(r, float) and isinstance(r1, float) or r == r1 == 2

        # Weights for measuring residual errors for 3 equations
        mu = self._constitutive_law.mu()
        # FIXME: Inf-sup constant of square; see Costabel for general polygons
        # https://perso.univ-rennes1.fr/martin.costabel/publis/Co_Mafelap2013_print.pdf
        beta = (0.5 - 1.0/pi)**0.5
        # FIXME: Lookup gamma computation in my notes; maybe 1 is correct
        gamma = 1.0

        # Compute estimators
        DG0 = FunctionSpace(mesh, 'DG', 0)
        v = TestFunction(DG0)
        eta_O_s = assemble(Constant((C_P/mu)**r1)*h**Constant(r1)*inner(f+div(q), f+div(q))**Constant(r1/2)*v*dx)
        eta_F_s = assemble(Constant(1.0/mu**r1)*inner(s-p*I-q, s-p*I-q)**Constant(r1/2)*v*dx)
        eta_D_r = assemble(Constant(1.0/beta**r)*(div(u)*div(u))**Constant(r/2)*v*dx)
        eta_I_r = assemble(Constant(1.0/gamma**r)*inner(dev(g(s, d, 0.0)), dev(g(s, d, 0.0)))**Constant(r/2)*v*dx)
        # TODO: Implement VecPow from PETSc
        # TODO: Avoid drastic numpy manipulations
        eta_O = eta_O_s.array() ** (1.0/r1)
        eta_F = eta_F_s.array() ** (1.0/r1)
        eta_D = eta_D_r.array() ** (1.0/r)
        eta_I = eta_I_r.array() ** (1.0/r)
        eta_1 = Function(DG0)
        eta_2 = Function(DG0)
        eta_3 = Function(DG0)
        eta_1.vector()[:] = eta_O + eta_F
        eta_2.vector()[:] = eta_D
        eta_3.vector()[:] = eta_I
        Eta_1 = MPI.sum(comm, np.sum(eta_1.vector().array() ** r1)) ** (1.0/r1)
        Eta_2 = MPI.sum(comm, np.sum(eta_2.vector().array() ** r)) ** (1.0/r)
        Eta_3 = MPI.sum(comm, np.sum(eta_3.vector().array() ** r)) ** (1.0/r)

        info_red("Estimators for ||R_1||, ||R_2||, ||R_3||: %g, %g, %g"
                 % (Eta_1, Eta_2, Eta_3))

        return Eta_1, Eta_2, Eta_3, eta_1, eta_2, eta_3


    def estimate_errors_components(self):
        w = self._w
        mesh = self._w.function_space().mesh()
        comm = mesh.mpi_comm()
        reconstructor = self.reconstructor()
        f = self._f
        r = self._constitutive_law.r()

        u, p, s = w.split()
        s = deviatoric(s)
        d = sym(grad(u))

        g = self._constitutive_law.g()
        eps = self._eps

        gdim = mesh.geometry().dim()

        # TODO: Consolidate notation
        q = []
        I = Identity(gdim)
        stress = -p*I + s
        algebraic_residual = 0.0
        for i in xrange(gdim):
            w = reconstructor.reconstruct(stress[i, :], f[i], algebraic_residual)
            q.append(w.sub(0, deepcopy=True))
        q = as_vector(q)

        # Poincare constant on cells
        cell_type = mesh.type()
        C_P = poincare_const(cell_type, r, gdim)
        h = CellDiameters(mesh)

        # Dual Lebesgue exponent; denoted s in the paper
        r1 = r/(r-1)
        # r/2 is dangerous for integer r != 2
        assert isinstance(r, float) and isinstance(r1, float) or r == r1 == 2

        # Weights for measuring residual errors for 3 equations
        mu = self._constitutive_law.mu()
        # FIXME: Inf-sup constant of square; see Costabel for general polygons
        # https://perso.univ-rennes1.fr/martin.costabel/publis/Co_Mafelap2013_print.pdf
        beta = (0.5 - 1.0/pi)**0.5
        # FIXME: Lookup gamma computation in my notes; maybe 1 is correct
        gamma = 1.0

        # Compute estimators
        DG0 = FunctionSpace(mesh, 'DG', 0)
        v = TestFunction(DG0)
        eta_disc = assemble((Constant(1.0/mu**r1)*inner(s-p*I-q, s-p*I-q)**Constant(r1/2)
                            +Constant(1.0/beta**r)*(div(u)*div(u))**Constant(r/2)
                            +Constant(1.0/gamma**r)*inner(dev(g(s, d, eps)), dev(g(s, d, eps)))**Constant(r/2))*v*dx)
        eta_reg = assemble(Constant(1.0/gamma**r)*inner(dev(g(s, d, eps)-g(s, d, 0.0)), dev(g(s, d, eps)-g(s, d, 0.0)))**Constant(r/2)*v*dx)
        eta_osc = assemble(Constant((C_P/mu)**r1)*h**Constant(r1)*inner(f+div(q), f+div(q))**Constant(r1/2)*v*dx)
        Eta_disc = eta_disc.norm('l1')
        Eta_reg = eta_reg.norm('l1')
        Eta_osc = eta_osc.norm('l1')

        info_red("Estimators eta_disc, eta_reg, eta_osc: %g, %g, %g"
                 % (Eta_disc, Eta_reg, Eta_osc))

        return Eta_disc, Eta_reg, Eta_osc, eta_disc, eta_reg, eta_osc


    def compute_exact_bounds(self, u_ex, p_ex, s_ex):
        w = self._w
        mesh = self._w.function_space().mesh()
        r = self._constitutive_law.r()

        u, p, s = w.split()
        s = deviatoric(s)

        # Dual Lebesgue exponent; denoted s in the paper
        r1 = r/(r-1)
        # r/2 is dangerous for integer r != 2
        assert isinstance(r, float) and isinstance(r1, float) or r == r1 == 2

        # Weight for residual norm for momentum equation
        mu = self._constitutive_law.mu()

        I = Identity(mesh.geometry().dim())
        DG0 = FunctionSpace(mesh, 'DG', 0)
        v = TestFunction(DG0)

        # Lower estimate
        lower = Function(DG0)
        normalization_factors = assemble(inner(grad(u-u_ex),
                                               grad(u-u_ex))
                                         **Constant(r/2) * v * dx)
        normalization_factor_global = mu*normalization_factors.sum()**(1.0/r)
        # TODO: Use VecPow, avoid numpy
        normalization_factors[:] = mu*normalization_factors.array()**(1.0/r)
        assemble(inner(s-p*I-(s_ex-p_ex*I), grad(u-u_ex))*v*dx,
                 tensor=lower.vector())
        Lower = lower.vector().sum()/normalization_factor_global
        as_backend_type(lower.vector()).vec().__idiv__(
                 as_backend_type(normalization_factors).vec())

        # Upper estimate
        upper = Function(DG0)
        assemble(inner(s-p*I-(s_ex-p_ex*I), s-p*I-(s_ex-p_ex*I))**Constant(r1/2)*v*dx,
                 tensor=upper.vector())
        Upper = upper.vector().sum()**Constant(1.0/r1) / mu
        # TODO: Use VecPow, avoid numpy
        upper.vector()[:] = upper.vector().array()**Constant(1.0/r1) / mu

        info_red("Bounds ||R_1||_L, ||R_1||_U: %g, %g" % (Lower, Upper))

        return Lower, Upper, lower, upper


    def compute_errors(self):
        mesh = self._w.function_space().mesh()

        # Grab approximate solution and rhs
        u_h, p_h, s_h = self._w.split()
        s_h = deviatoric(s_h)
        I = Identity(s_h.ufl_shape[0])
        f = self.f

        # Grab constitutive parameters
        r = self._constitutive_law.r()

        # Weights for measuring residual errors for 3 equations
        mu = self._constitutive_law.mu()

        if r != 2:
            warning("Momentum resdidual lifting only implemented for r == 2.")
            return None, None

        # Use large degree for lifting residual_1 to H^1_0 to reduce
        # discretization error
        u_h_degree = u_h.function_space().ufl_element().degree()
        f_degree = f.ufl_element().degree()
        # TODO: Review this! f_degree=6 !
        lifting_1_degree = max(u_h_degree + 2, f_degree)
        lifting_1_space = VectorFunctionSpace(mesh, "Lagrange", lifting_1_degree)
        res_1, v = TrialFunction(lifting_1_space), TestFunction(lifting_1_space)
        residual_1 = ( dot(f, v) - inner(s_h - p_h*I, grad(v)) )*dx
        a_1 = inner(grad(res_1), grad(v))*dx
        res_1 = Function(lifting_1_space)
        bc_1 = DirichletBC(lifting_1_space, res_1.value_dimension(0)*(0.0,), "on_boundary")
        solve(a_1 == residual_1, res_1, bcs=bc_1)
        Res_1 = 1.0/mu*norm(res_1, norm_type="H10")

        info_red("||R_1||_lifting = %g" % Res_1)

        # FIXME: res_1 is wrongly scaled!
        #return Res_1, res_1
        return Res_1, None
