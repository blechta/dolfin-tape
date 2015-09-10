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

"""This script solves p-Laplace problem and shows how different components of
residual-based equilibrated flux reconstruction error estimate behave.

TODO: Add relevant reference.
"""

from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

from dolfintape import FluxReconstructor, CellDiameters


mesh = UnitSquareMesh(40, 40, 'crossed')
V = FunctionSpace(mesh, 'Lagrange', 1)
DG0 = FunctionSpace(mesh, 'DG', 0)
f = Expression("1.+cos(2.*pi*x[0])*sin(2.*pi*x[1])", domain=mesh, degree=2)

# Initialize mesh reconstructior once - the system matrix is the same
tic()
reconstructor = FluxReconstructor(mesh, 1)
info_green('Flux reconstructor initialization timing: %g seconds' % toc())


def solve_p_laplace(p, eps, u0=None):
    p1 = p/(p-1) # Dual Lebesgue exponent
    p = Constant(p)
    p1 = Constant(p1)
    eps = Constant(eps)

    # Initial approximation for Newton
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
    S_eps = (eps + inner(grad(u), grad(u)))**(0.5*p-1.0) * grad(u)
    tic()
    q = reconstructor.reconstruct(S, f).sub(0, deepcopy=False)
    info_green('Flux reconstruction timing: %g seconds' % toc())

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


def plot_conv(p, errors):
    errors = np.array(errors, dtype='float')
    plt.plot(errors[:, 0], errors[:, 3], 'o-', label='$p=%g$ overall'%p)
    plt.plot(errors[:, 0], errors[:, 1], 'o-', label='$p=%g$ discretization'%p)
    plt.plot(errors[:, 0], errors[:, 2], 'o-', label='$p=%g$ regularization'%p)
    plt.title(r'Error estimates of '
              r'$||f+\mathrm{div}|\nabla u|^{p-2}\nabla u||_{-1,p}$')
    plt.xlabel(r'$\epsilon$')
    plt.loglog()


def run_demo(p, epsilons, zero_guess=False):
    file = XDMFFile(mesh.mpi_comm(), 'results/p%g.xdmf' % p)
    estimates = []
    u = Function(V)
    for eps in epsilons:
        if zero_guess:
            u.vector().zero()
        u, est_h, est_eps, est_tot = solve_p_laplace(p, eps, u)
        estimates.append((eps, est_h, est_eps, est_tot))
        if MPI.size(mesh.mpi_comm()) == 1:
            file << (u, eps) # FIXME: Deadlock in parallel! Why?
        plot(u, title='p-Laplace, p=%g, eps=%g'%(p, eps))
    plot_conv(p, estimates)

    plt.legend(loc=4)
    plt.savefig('results/convergence_p%g.pdf' % p)
    plt.show(block=True)
    interactive()


if __name__ == '__main__':

    # p = 11.0
    run_demo(11.0, [10.0**i for i in np.arange(1.0,  -6.0, -0.5)])

    # p = 1.1; works better with zero guess
    run_demo( 1.1, [10.0**i for i in np.arange(0.0, -22.0, -2.0)], True)
