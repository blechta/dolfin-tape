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

import unittest
from dolfin import *

from dolfintape import mesh_diameter


class TestCase(unittest.TestCase):

    if MPI.size(mpi_comm_world()) == 1:
        def test_mesh_diameter(self):

            mesh = UnitSquareMesh(10, 7)
            d = mesh_diameter(mesh)
            self.assertAlmostEqual(d, 2.0**0.5)

            mesh = UnitCubeMesh(3, 4, 5)
            d = mesh_diameter(mesh)
            self.assertAlmostEqual(d, 3.0**0.5)

            try:
                import mshr
            except ImportError:
                warning("mshr not available; skipping some tests of "
                        "'mesh_diameter'...")
            else:

                b0 = mshr.Rectangle(Point(0.0, 0.0), Point(0.5, 1.0))
                b1 = mshr.Rectangle(Point(0.0, 0.0), Point(1.0, 0.5))
                lshape = b0 + b1
                mesh = mshr.generate_mesh(lshape, 10)
                d = mesh_diameter(mesh)
                self.assertAlmostEqual(d, 2.0**0.5)

                s = mshr.Sphere(Point(100.0, -666666.6, 1e10), 4.0, 5)
                mesh = mshr.generate_mesh(s, 5)
                d = mesh_diameter(mesh)
                self.assertAlmostEqual(d, 8.0, 0)


    if MPI.size(mpi_comm_world()) > 1:
        def test_mesh_diameter_notimplemeted(self):
            mesh = UnitIntervalMesh(10)
            with self.assertRaises(RuntimeError):
                mesh_diameter(mesh)


if __name__ == "__main__":
    unittest.main()
