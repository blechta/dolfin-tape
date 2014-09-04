import unittest
from dolfin import *
import numpy as np
import random

from common import MatrixView, VectorView

m, n = 2000, 300
mesh = UnitSquareMesh(m, n)
V = FunctionSpace(mesh, 'CG', 1)
u, v = TrialFunction(V), TestFunction(V)
a = inner(grad(u), grad(v))*dx
L = v*dx

class MatrixViewTest(unittest.TestCase):

    def test_full_view(self):

        A = Matrix()
        AssemblerBase().init_global_tensor(A, Form(a))

        tic()
        assemble(a, tensor=A)
        t_assemble = toc()

        # Number DOFs including ghosts; Lagrange deg 1 assumend
        num_dofs = mesh.num_vertices()
        ind = np.arange(num_dofs, dtype='uintp')

        A1 = A.copy()
        A1.zero()

        tic()
        B = MatrixView(A1, V.dim(), V.dim(), ind, ind)
        t_matview_constructor = toc()
        self.assertLess(t_matview_constructor, 0.5)
        del ind # Check that ind is not garbage collected prematurely

        tic()
        assemble(a, tensor=B, add_values=True)
        t_assemble_matview = toc()
        self.assertLess(t_assemble_matview, 2.0*t_assemble)

        print 'Timings:'
        print '  Regular assemble                 ', t_assemble
        print '  Assemble into MatrixView         ', t_assemble_matview

        A1 -= A
        errornorm = A1.norm('linf')
        self.assertAlmostEqual(errornorm, 0.0)


class VectorViewTest(unittest.TestCase):

    def test_full_view(self):

        x = Vector()
        AssemblerBase().init_global_tensor(x, Form(L))

        tic()
        assemble(L, tensor=x)
        t_assemble = toc()

        # Number DOFs including ghosts; Lagrange deg 1 assumend
        num_dofs = mesh.num_vertices()
        ind = np.arange(num_dofs, dtype='uintp')

        x1 = x.copy()
        x1.zero()

        tic()
        y = VectorView(x1, V.dim(), ind)
        t_vecview_constructor = toc()
        self.assertLess(t_vecview_constructor, 0.5)

        tic()
        assemble(L, tensor=y, add_values=True)
        t_assemble_vecview = toc()
        self.assertLess(t_assemble_vecview, 2.0*t_assemble)

        x1 -= x
        errornorm = x1.norm('linf')
        self.assertAlmostEqual(errornorm, 0.0)

        # Shuffle DOFs; owned and ghosts separately
        range = V.dofmap().ownership_range()
        num_owned_dofs = range[1] - range[0]
        random.seed(42)
        random.shuffle(ind[:num_owned_dofs ])
        random.shuffle(ind[ num_owned_dofs:])

        x1.zero() # NOTE: VectorView.zero,norm not yet implemented, so calling to x1
        tic()
        assemble(L, tensor=y, add_values=True)
        t_assemble_vecview_shuffled = toc()

        # Compare norms
        self.assertAlmostEqual(x.norm('l1'), x1.norm('l1'))

        # Check that arrays are the same
        # NOTE: Works only sequentially as we did not shuffle ghosts properly
        if MPI.size(mpi_comm_world()) == 1:
            self.assertAlmostEqual(sum(abs(x1.array()[ind[:num_owned_dofs]]-x.array())), 0.0)

        print 'Timings:'
        print '  Regular assemble                 ', t_assemble
        print '  Assemble into VectorView         ', t_assemble_vecview
        print '  Assemble into shuffled VectorView', t_assemble_vecview_shuffled


if __name__ == "__main__":
    unittest.main()
