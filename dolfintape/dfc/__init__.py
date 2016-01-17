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

"""This module implements disk-free caching FEniCS form compiler.
JITting of Forms and FiniteElements is supported. Support for DOLFIN
Subdomains and Expression is missing.

Importing this module injects functions 'jit' and 'default_parameters'
into 'dolfintape' namespace. Then 'dolfintape' package can be used as a
FEniCS form compiler in DOLFIN. Usage:

    import dolfintape.dfc
    import dolfin

    dolfin.parameters["form_compiler"]["name"] = "dolfintape"
    dolfin.parameters["form_compiler"]["spam"] = "eggs"
    ...

NOTE: There is already memory caching mechanism in instant.cache.
      This one is only slightly faster than the one in instant.
      Moreover instant version works also for DOLFIN JIT code.
FIXME: Consider removing this module.
"""

import dolfintape
from dolfintape.dfc.jitcompiler import jit, default_parameters
import dolfintape.dfc.cache

dolfintape.jit = jit
dolfintape.default_parameters = default_parameters
