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

from sympy import Symbol, symbols, sin, pi, Matrix, diff, integrate
from sympy import ccode as sympy_ccode
from dolfin import Expression

__all__ = ["pLaplace_modes", "pStokes_vortices"]


ccode = lambda *args, **kwargs: sympy_ccode(*args, **kwargs).replace('M_PI', 'pi')


def pLaplace_modes(*args, **kwargs):
    """Returns 2-tuple of DOLFIN Expressions initialized with *args and
    **kwargs passed in and solving zero BC p-Laplace problem on unit square
    as solution and corresponding right-hand side.

    Mandatory kwargs:
      kwargs['p'] > 1.0    ... Lebesgue exponent
      kwargs['eps'] >= 0.0 ... amount of regularization
      kwargs['n'] uint     ... x-mode
      kwargs['m'] uint     ... y-mode
    """
    p = Symbol('p', positive=True, constant=True)
    eps = Symbol('eps', nonnegative=True, constant=True)
    n = Symbol('n', integer=True, positive=True, constant=True)
    m = Symbol('m', integer=True, positive=True, constant=True)
    x = symbols('x[0] x[1]', real=True)
    dim = len(x)
    u = sin(n*pi*x[0])*sin(m*pi*x[1])
    Du = Matrix([diff(u, x[j]) for j in xrange(dim)])
    q = (eps + Du.norm(2)**2)**(p/2-1) * Du
    f = -sum(diff(q[j], x[j]) for j in xrange(dim))

    u_code = ccode(u)
    # Prevent division by zero
    f_code = "x[0]*x[0] + x[1]*x[1] < 1e-308 ? 0.0 : \n" + ccode(f)
    return [Expression(u_code, *args, **kwargs),
            Expression(f_code, *args, **kwargs)]


def pStokes_vortices(*args, **kwargs):
    """Returns 4-tuple of DOLFIN Expressions initialized with *args and
    **kwargs passed in and solving no-slip p-Stokes problem on unit square
    as velocity, pressure, extra stress and body force.

    Mandatory kwargs:
      kwargs['eps'] >= 0.0 ... amount of regularization
      kwargs['n'] uint     ... number of vortices in each direction
      kwargs['mu'] > 0.0   ... 'viscosity'
    Optional kwargs:
      kwargs['r'] > 1.0    ... power-law exponent, default 2
    """
    if kwargs.get('r', 2) == 2:
        codes = _pStokes_vortices_ccode(r=2)
    else:
        codes = _pStokes_vortices_ccode()
    return (Expression(c, *args, **kwargs) for c in codes)


def _pStokes_vortices_ccode(r=None):

    n = Symbol('n', integer=True, positive=True, constant=True)
    mu = Symbol('mu', positive=True, constant=True)
    x = symbols('x[0] x[1]')
    dim = len(x)
    u = (sin(n*pi*x[0])**2*sin(2*n*pi*x[1]), -sin(n*pi*x[1])**2*sin(2*n*pi*x[0]))
    p = x[0]**2
    p = p - integrate(p, (x[0], 0, 1), (x[1], 0, 1))

    L = Matrix(dim, dim, [diff(u[i], x[j]) for i in xrange(dim) for j in xrange(dim)])
    D = (L+L.T)/2
    D2 = (D*D.T).trace()
    eps = Symbol('eps', nonnegative=True, constant=True)
    if not r:
        r = Symbol('r', positive=True, constant=True)
    S = 2*mu*(eps + D2)**(r/2-1)*D
    divS = tuple(sum(diff(S[i, j], x[j]) for j in xrange(dim)) for i in xrange(dim))
    gradp = tuple(diff(p, x[i]) for i in xrange(dim))
    f = tuple(gradp[i] - divS[i] for i in xrange(dim))

    p_code = ccode(p)
    u_code = tuple(ccode(u[i]) for i in xrange(dim))
    f_code = tuple(ccode(f[i]) for i in xrange(dim))
    S_code = tuple(tuple(ccode(S[i, j]) for j in xrange(dim)) for i in xrange(dim))

    return u_code, p_code, S_code, f_code
