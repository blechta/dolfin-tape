from dolfin import *

from common.deviatoric_space import TensorFunctionSpace, deviatoric
from problems.GeneralizedStokes import GeneralizedStokesProblem

__all__ = ['StokesVortices']


class NewtonianFluid(object):
    def __init__(self, mu):
        self._mu = mu

    def r(self):
        return 2

    def mu(self):
        return self._mu

    def g(self):
        return lambda s, d: s - 2.0*Constant(self._mu)*d


class StokesVortices(GeneralizedStokesProblem):
    n = 4 # Number of vortices
    mu = 1.0
    u_ex = Expression(('+pow(sin(n*pi*x[0]), 2) * sin(2.0*n*pi*x[1])',
                       '-pow(sin(n*pi*x[1]), 2) * sin(2.0*n*pi*x[0])'),
                      n=n, degree=6)
    p_ex = Expression('0.0')
    s_ex = 2.0*Constant(mu)*Expression(
            (('n*pi*sin(2.0*n*pi*x[0])*sin(2.0*n*pi*x[1])',
              'n*pi*( pow(sin(n*pi*x[0]), 2)*cos(2.0*n*pi*x[1])  '
              '     - pow(sin(n*pi*x[1]), 2)*cos(2.0*n*pi*x[0]) )'),
             ('n*pi*( pow(sin(n*pi*x[0]), 2)*cos(2.0*n*pi*x[1])  '
              '     - pow(sin(n*pi*x[1]), 2)*cos(2.0*n*pi*x[0]) )',
              '-n*pi*sin(2.0*n*pi*x[0])*sin(2.0*n*pi*x[1])')),
             n=n, degree=6)
    f = Constant(mu)* \
        Expression(('+2.0*n*n*pi*pi*( 2.0*pow(sin(n*pi*x[0]), 2) - cos(2.0*n*pi*x[0]) ) * sin(2.0*n*pi*x[1])',
                    '-2.0*n*n*pi*pi*( 2.0*pow(sin(n*pi*x[1]), 2) - cos(2.0*n*pi*x[1]) ) * sin(2.0*n*pi*x[0])'),
                   n=n, degree=6)

    def __init__(self, N):
        mesh = UnitSquareMesh(N, N, "crossed")
        constitutive_law = NewtonianFluid(self.mu)
        GeneralizedStokesProblem.__init__(self, mesh, constitutive_law, self.f)

    def compute_errors(self):
        mesh = self._w.function_space().mesh()

        # Grab approximate solution and rhs
        u_h, p_h, s_h = self._w.split()
        s_h = deviatoric(s_h)
        I = Identity(s_h.shape()[0])
        f = self.f

        # Grab constitutive parameters
        g = self._constitutive_law.g()
        r = self._constitutive_law.r()
        mu = self._constitutive_law.mu()

        # Weights for measuring residual errors for 3 equations
        alpha = (2.0*float(mu))**(1.0/r)
        # FIXME: Obtain Brezzi constants by eigensolver
        beta = Constant(0.1)
        gamma = Constant(0.1)

        # Use large degree for lifting residual_1 to H^1_0 to reduce
        # discretization error
        # NOTE: residual_3 may have error or not; depending whether g
        #       is polynomial or not
        u_h_degree = u_h.function_space().ufl_element().degree()
        s_h_degree = s_h.operands()[0].operands()[0].operands()[0] \
                        .function_space().ufl_element().degree()
        f_degree = self.f.ufl_element().degree()
        lifting_1_degree = max(u_h_degree + 2, f_degree) # TODO: Review this! f_degree=6 !
        lifting_2_degree = u_h_degree - 1
        # NOTE: For Newtonian fluid this is exact, without discretization error
        lifting_3_degree = max(s_h_degree, u_h_degree - 1)
        info_blue("lifting degrees %g, %g, %g"
                  % (lifting_1_degree, lifting_2_degree, lifting_3_degree))
        lifting_1_space = VectorFunctionSpace(mesh, "Lagrange", lifting_1_degree)
        lifting_2_space = FunctionSpace(mesh, "Discontinuous Lagrange", lifting_2_degree)
        lifting_3_space = TensorFunctionSpace(mesh, "Discontinuous Lagrange", lifting_3_degree,
                                              symmetry=True, zero_trace=True)
        res_1, v = TrialFunction(lifting_1_space), TestFunction(lifting_1_space)
        res_2, q = TrialFunction(lifting_2_space), TestFunction(lifting_2_space)
        res_3, t = TrialFunction(lifting_3_space), TestFunction(lifting_3_space)
        res_3, t = deviatoric(res_3), deviatoric(t)
        # FIXME: Which way is better? Former does not need exact solutions at all!
        #residual_1 = ( dot(f, v) - inner(s_h - p_h*I, grad(v)) )*dx
        residual_1 = (  inner(self.s_ex - self.p_ex*I, grad(v)) - inner(s_h - p_h*I, grad(v)) )*dx
        residual_2 = -div(u_h)*q*dx
        residual_3 = -inner(dev(g(s_h, sym(grad(u_h)))), t)*dx
        a_1 = inner(grad(res_1), grad(v))*dx
        a_2 = res_2*q*dx
        a_3 = inner(res_3, t)*dx
        res_1 = Function(lifting_1_space)
        res_2 = Function(lifting_2_space)
        res_3 = Function(lifting_3_space)
        bc_1 = DirichletBC(lifting_1_space, res_1.value_dimension(0)*(0.0,), "on_boundary")
        solve(a_1 == residual_1, res_1, bcs=bc_1)
        solve(a_2 == residual_2, res_2)
        solve(a_3 == residual_3, res_3)
        res_3 = deviatoric(res_3)
        Res_1 = 1.0/alpha*norm(res_1, norm_type="H10")
        Res_2 = 1.0/beta*norm(res_2, norm_type="L2")
        # FIXME: This will not work as norm accepts only GenericFunction / GenericVector
        #Res_3 = 1.0/gamma*norm(res_3, norm_type="L2")
        Res_3 = 1.0/gamma*assemble(inner(res_3, res_3)*dx)**0.5

        info_red("Residual norms seem to be %g, %g, %g"
                 % (Res_1, Res_2, Res_3))

        return Res_1, Res_2, Res_3, res_1, res_2, res_3
