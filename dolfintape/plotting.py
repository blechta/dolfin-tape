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
import numpy as np
from six.moves import xrange as range

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

__all__ = ["plot_alongside"]

parameters['plotting_backend'] = 'matplotlib' # FIXME: side effect!


def plot_alongside(*args, **kwargs):
    """Plot supplied functions in single figure with common colorbar.
    User may supply 'range_min' and 'range_max' in kwargs.
    """
    if not kwargs.has_key("range_min") or not kwargs.has_key("range_max"):
        # Look for common range of plot
        m, M = np.inf, -np.inf
        for f in args:
            assert isinstance(f, Function)
            assert f.ufl_shape == ()
            assert f.ufl_element().family() in ["Lagrange", "Discontinuous Lagrange"]
            if f.ufl_element().degree() > 1:
                warning("Search for min/max of Lagrange function of degree > 1"
                        " may not be accurate")
            m, M = min(m, f.vector().min()), max(M, f.vector().max())
        if not kwargs.has_key("range_min"):
            kwargs["range_min"] = m
        if not kwargs.has_key("range_max"):
            kwargs["range_max"] = M

    # Prepare figure
    n = len(args)
    plt.figure(figsize=(4*n+2, 4))
    projection = "3d" if kwargs.get("mode") == "warp" else None

    # Plot all arguments
    for i in range(n):
        plt.subplot(1, n, i+1, projection=projection)
        p = plot(args[i], **kwargs)

    plt.tight_layout()

    # Create colorbar
    plt.subplots_adjust(right=0.8)
    cbar_ax = plt.gcf().add_axes([0.85, 0.15, 0.05, 0.7])
    plt.colorbar(p, cax=cbar_ax)
