from dolfin import *
import ufl
ufl.set_level(ufl.INFO) # Enable info_{green,red,blue}

import matplotlib.pyplot as plt
import numpy as np

from common import FluxReconstructor
from common.deviatoric_space import TensorFunctionSpace, deviatoric
from common.cell_diameter import CellDiameters


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

        # Add some diagonal zeros to make PETSc happy; this could be solved
        # also with Assembler option keep_diagonal
        F += Constant(0.0)*p*q*dx

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

        # Compure estimators
        DG0 = FunctionSpace(mesh, 'DG', 0)
        v = TestFunction(DG0)
        eta_R_s = assemble(Constant(C_P**r1/alpha**r1)*h*inner(f+div(q), f+div(q))**(r1/2)*v*dx)
        eta_F_s = assemble(Constant(1.0/alpha**r1)*inner(s-p*I-q, s-p*I-q)**(r1/2)*v*dx)
        eta_D_r = assemble(Constant(1.0/beta**r)*div(u)**r*v*dx)
        # FIXME: Fix eta_I (4.3d) in paper =- there should be deviatoric part of g
        #eta_I_r = assemble(Constant(1.0/gamma**r)*inner(g(s, d), g(s, d))**(r/2)*v*dx)
        eta_I_r = assemble(Constant(1.0/gamma**r)*inner(dev(g(s, d)), dev(g(s, d)))**(r/2)*v*dx)
        # TODO: Implement VecPow from PETScA
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
        Eta_1 = np.sum(eta_1.vector().array() ** r1) ** (1.0/r1)
        Eta_2 = np.sum(eta_2.vector().array() ** r) ** (1.0/r)
        Eta_3 = np.sum(eta_3.vector().array() ** r) ** (1.0/r)

        info_red("Estimators for ||R_1||, ||R_2||, ||R_3||: %g, %g, %g"
                 % (Eta_1, Eta_2, Eta_3))

        return Eta_1, Eta_2, Eta_3, eta_1, eta_2, eta_3