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
        weight = Constant((2.0*mu)**(-1.0/r))
        I_RTN = lambda x: x # Dummy interpolator
        return ( weight*I_RTN(hat[p]*inner(stress_component, v))
                 -hat[p]*force_component*phi
                 +inner(grad(hat[p]), stress_component)*phi
                 +residual_component*phi )*self.dp(p)

    def a(self, p):
        """lhs of (6.7); one component"""
        d, q = TrialFunctions(self._W)
        v, phi = TestFunctions(self._W)
        mu, r = float(self._mu), float(self._r)
        weight = Constant((2.0*mu)**(-1.0/r))
        return ( weight*inner(d, v) + inner(q, div(v)) + inner(phi, div(d)) )*self.dp(p)


class GeneralizedStokesProblem(object):

    def __init__(self, mesh, constitutive_law, f):

        V = VectorFunctionSpace(mesh, 'CG', 2)
        Q = FunctionSpace(mesh, 'CG', 1)
        S = TensorFunctionSpace(mesh, 'DG', 1, symmetry=True, zero_trace=True)
        W = MixedFunctionSpace([V, Q, S])

        w = Function(W)
        (u, p, s) = split(w)
        (v, q, t) = TestFunctions(W)
        s = deviatoric(s)
        t = deviatoric(t)

        d = sym(grad(u))
        g = constitutive_law.g()
        F = ( inner(s, grad(v)) - p*div(v) - q*div(u) + inner(g(s, d), t) - inner(f, v) )*dx

        bc_u = DirichletBC(W.sub(0), (0.0, 0.0), lambda x, b: b)
        bc_p = DirichletBC(W.sub(1), 0.0, "near(x[0], 0.0) && near(x[1], 0.0)", method="pointwise")

        self._F = F
        self._J = derivative(self._F, w)
        self._bcs = [bc_u, bc_p]
        self._w = w
        self._constitutive_law = constitutive_law
        self._f = f


    def solve(self):
        solve(self._F == 0, self._w, bcs=self._bcs, J=self._J)
        return self._w


    def reconstructor(self):
        try:
            reconstructor = self._reconstructor
        except AttributeError:
            mesh = self._w.function_space().mesh()
            l = self._w.function_space().ufl_element().sub_elements()[0].degree()
            r = self._constitutive_law.r()
            mu = self._constitutive_law.mu()
            reconstructor = ImplicitStokesFluxReconstructor(mesh, l, r, mu)
            self._reconstructor = reconstructor
        return reconstructor


    def estimate_errors(self):
        w = self._w
        mesh = self._w.function_space().mesh()
        comm = mesh.mpi_comm()
        reconstructor = self.reconstructor()
        f = self._f
        r = self._constitutive_law.r()
        mu = self._constitutive_law.mu()

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

        # Cell diameter
        h = CellDiameters(mesh)

        # Poincare constant on simplex
        # FIXME: Add Poincare constant for r != 2
        assert r == 2, "Poincare constant not implemented for r != 2"
        C_P = 1.0/pi

        # Dual Lebesgue exponent; denoted s in the paper
        r1 = r/(r-1)
        # r/2 is dangerous for integer r != 2
        assert isinstance(r, float) and isinstance(r1, float) or r == r1 == 2
        # FIXME: Wrap float r, r1 by Constant to avoid code generation bloat

        # Weights for measuring residual errors for 3 equations
        alpha = (2.0*float(mu))**(1.0/r)
        # FIXME: Obtain Brezzi constants by eigensolver
        beta = Constant(0.1)
        gamma = Constant(0.1)

        # Compute estimators
        DG0 = FunctionSpace(mesh, 'DG', 0)
        v = TestFunction(DG0)
        eta_R_s = assemble(Constant(C_P**r1/alpha**r1)*h*inner(f+div(q), f+div(q))**(r1/2)*v*dx)
        eta_F_s = assemble(Constant(1.0/alpha**r1)*inner(s-p*I-q, s-p*I-q)**(r1/2)*v*dx)
        eta_D_r = assemble(Constant(1.0/beta**r)*div(u)**r*v*dx)
        eta_I_r = assemble(Constant(1.0/gamma**r)*inner(dev(g(s, d)), dev(g(s, d)))**(r/2)*v*dx)
        # TODO: Implement VecPow from PETSc
        # TODO: Avoid drastic numpy manipulations
        eta_R = eta_R_s.array() ** (1.0/r1)
        eta_F = eta_F_s.array() ** (1.0/r1)
        eta_D = eta_D_r.array() ** (1.0/r)
        eta_I = eta_I_r.array() ** (1.0/r)
        eta_1 = Function(DG0)
        eta_2 = Function(DG0)
        eta_3 = Function(DG0)
        eta_1.vector()[:] = eta_R + eta_F
        eta_2.vector()[:] = eta_D
        eta_3.vector()[:] = eta_I
        Eta_1 = MPI.sum(comm, np.sum(eta_1.vector().array() ** r1)) ** (1.0/r1)
        Eta_2 = MPI.sum(comm, np.sum(eta_2.vector().array() ** r)) ** (1.0/r)
        Eta_3 = MPI.sum(comm, np.sum(eta_3.vector().array() ** r)) ** (1.0/r)

        info_red("Estimators for ||R_1||, ||R_2||, ||R_3||: %g, %g, %g"
                 % (Eta_1, Eta_2, Eta_3))

        return Eta_1, Eta_2, Eta_3, eta_1, eta_2, eta_3


    def compute_exact_bounds(self, u_ex, p_ex, s_ex):
        w = self._w
        mesh = self._w.function_space().mesh()
        r = self._constitutive_law.r()
        mu = self._constitutive_law.mu()

        u, p, s = w.split()
        s = deviatoric(s)

        # Dual Lebesgue exponent; denoted s in the paper
        r1 = r/(r-1)
        # r/2 is dangerous for integer r != 2
        assert isinstance(r, float) and isinstance(r1, float) or r == r1 == 2
        # FIXME: Wrap float r, r1 by Constant to avoid code generation bloat

        # Weight for residual norm for momentum equation
        alpha = (2.0*float(mu))**(1.0/r)

        I = Identity(mesh.geometry().dim())
        DG0 = FunctionSpace(mesh, 'DG', 0)
        v = TestFunction(DG0)

        # Lower estimate
        lower = Function(DG0)
        normalization_factors = assemble(inner(grad(u-u_ex),
                                               grad(u-u_ex))
                                         **(r/2) * v * dx)
        normalization_factor_global = alpha*normalization_factors.sum()**(1.0/r)
        # TODO: Use VecPow, avoid numpy
        normalization_factors[:] = alpha*normalization_factors.array()**(1.0/r)
        assemble(inner(s-p*I-(s_ex-p_ex*I), grad(u-u_ex))*v*dx,
                 tensor=lower.vector())
        Lower = lower.vector().sum()/normalization_factor_global
        as_backend_type(lower.vector()).vec().__idiv__(
                 as_backend_type(normalization_factors).vec())

        # Upper estimate
        upper = Function(DG0)
        assemble(inner(s-p*I-(s_ex-p_ex*I), s-p*I-(s_ex-p_ex*I))**(r1/2)*v*dx,
                 tensor=upper.vector())
        Upper = upper.vector().sum()**(1.0/r1) / alpha
        # TODO: Use VecPow, avoid numpy
        upper.vector()[:] = upper.vector().array()**(1.0/r1) / alpha

        info_red("Bounds ||R_1||_L, ||R_1||_U: %g, %g" % (Lower, Upper))

        return Lower, Upper, lower, upper


    def compute_errors(self):
        mesh = self._w.function_space().mesh()

        # Grab approximate solution and rhs
        u_h, p_h, s_h = self._w.split()
        s_h = deviatoric(s_h)
        I = Identity(s_h.shape()[0])
        f = self.f

        # Grab constitutive parameters
        r = self._constitutive_law.r()
        mu = self._constitutive_law.mu()

        # Weights for measuring residual errors for 3 equations
        alpha = (2.0*float(mu))**(1.0/r)

        if r != 2:
            raise NotImplementedError("Momentum resdidual lifting only "
            "implemented for p == 2.")

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
        Res_1 = 1.0/alpha*norm(res_1, norm_type="H10")

        info_red("||R_1||_lifting = %g" % Res_1)

        # FIXME: res_1 is wrongly scaled!
        #return Res_1, res_1
        return Res_1, None
