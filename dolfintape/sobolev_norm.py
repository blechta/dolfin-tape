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

from dolfin import Constant, dx, grad, inner, assemble
import numpy as np

__all__ = ['sobolev_norm']


def sobolev_norm(u, p, k=1, domain=None):
    """Return Sobolev seminorm on W^{k, p} of function u. If u is
    None, return infinity."""
    # Special case
    if u is None:
        return np.infty

    # Prepare exponent and measure
    p = Constant(p) if p is not 2 else p
    dX = dx(domain)

    # Take derivative if k==1
    if k == 1:
        u = grad(u)
    elif k == 0:
        u = u
    else:
        raise NotImplementedError

    # Assemble functional and return
    return assemble(inner(u, u)**(p/2)*dX)**(1.0/float(p))
