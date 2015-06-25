from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

from common import FluxReconstructor
from common.cell_diameter import CellDiameters


class NonlinearFluxReconstructor(FluxReconstructor):
    def L(self, p, S, f):
        """Returns rhs linear form for flux reconstruction on patches p s.t.
        resulting flux q \in H^p1(div) fulfills
          * reconstructs q ~ -S
          * equilibrates div(q) ~ f."""
        v, q = TestFunctions(self._W)
        hat = self._hat
        return ( -hat[p]*inner(S, v)
                 -hat[p]*f*q
                 +inner(grad(hat[p]), S)*q )*self.dp(p)


mesh = UnitSquareMesh(40, 40, 'crossed')
V = FunctionSpace(mesh, 'Lagrange', 1)
DG0 = FunctionSpace(mesh, 'DG', 0)
f = Expression("1.+cos(2.*pi*x[0])*sin(2.*pi*x[1])", domain=mesh, degree=2)

# Initialize mesh reconstructior once - the system matrix is the same
tic()
reconstructor = NonlinearFluxReconstructor(mesh, 1)
info_green('Flux reconstructor initialization timing: %g seconds' % toc())


def solve_p_laplace(p, eps, u0=None):
    p1 = p/(p-1) # Dual Lebesgue exponent
    p = Constant(p)
    p1 = Constant(p1)
    eps = Constant(eps)

    # Initial approximation for Newton
    #u = u0.copy(deepcopy=True) if u0 else Function(V)
    u = u0 if u0 else Function(V)

    # Problem formulation
    E = ( 1./p*(eps + dot(grad(u), grad(u)))**(0.5*p) - f*u ) * dx
    F = derivative(E, u)
    bc = DirichletBC(V, 0.0, lambda x, onb: onb)
    solve(F == 0, u, bc)

    # Reconstruct flux q in H^p1(div) s.t.
    #       q ~ -S
    #   div q ~ f
    S = inner(grad(u), grad(u))**(0.5*p-1.0) * grad(u)
    tic()
    q = reconstructor.reconstruct(S, f).sub(0, deepcopy=False)
    info_green('Flux reconstruction timing: %g seconds' % toc())

    # Compute error estimate using equilibrated stress reconstruction
    v = TestFunction(DG0)
    h = CellDiameters(mesh)
    Cp = Constant(2.0*(0.5*p)**(1.0/p)) # Poincare estimates by [Chua, Wheeden 2006]
    err0 = assemble(inner(S+q, S+q)**(0.5*p1)*v*dx)
    err1 = assemble(((Cp*h*(f-div(q)))**2)**(0.5*p1)*v*dx)
    p1 = float(p1)
    err0[:] = err0.array()**(1.0/p1) + err1.array()**(1.0/p1)
    err_est = MPI.sum( mesh.mpi_comm(),
                       (err0.array().__abs__()**p1).sum() )**(1.0/p1)
    info_red('Error estimate %g' % err_est)

    return u, err_est


def plot_conv(p, errors):
    errors = np.array(errors, dtype='float')
    plt.plot(errors[:, 0], errors[:, 1], 'o-', label='$p=%g$' % p)
    plt.title(r'Error estimate of '
              r'$||f+\mathrm{div}|\nabla u|^\frac{p-2}{2}\nabla u||_{-1,p}$')
    plt.xlabel(r'$\epsilon$')
    plt.loglog()


if __name__ == '__main__':
    p = 11.0
    epsilons = [10.0**i for i in np.arange(1.0, -6.0, -0.5)]
    estimates = []
    u = None
    for eps in epsilons:
        u, est = solve_p_laplace(p, eps, u)
        estimates.append((eps, est))
    plot(u, title='p-Laplace, p=%g, eps=%g'%(p, eps))
    plot_conv(p, estimates)

    p = 1.1
    epsilons = [10.0**i for i in range(-10, -22, -2)]
    estimates = []
    for eps in epsilons:
        u, est = solve_p_laplace(p, eps)
        estimates.append((eps, est))
    plot(u, title='p-Laplace, p=%g, eps=%g'%(p, eps))
    plot_conv(p, estimates)

    plt.legend()
    plt.show(block=True)
    interactive()
