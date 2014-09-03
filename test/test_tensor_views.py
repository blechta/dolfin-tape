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

        # Upper bound on dof count (including ghosts)
        num_dofs = V.dofmap().max_cell_dimension()*mesh.num_cells()
        ind = np.arange(num_dofs, dtype='uintp')

        A1 = A.copy()
        A1.zero()

        tic()
        B = MatrixView(A1, V.dim(), V.dim(), ind, ind)
        del ind # Check that ind is not garbage collected prematuraly
        t_matview_constructor = toc()
        self.assertLess(t_matview_constructor, 0.5)

        tic()
        assemble(a, tensor=B, add_values=True)
        t_assemble_matview = toc()
        self.assertLess(t_assemble_matview, 2.0*t_assemble)

        print 'Timings:'
        print '  Regular assemble        ', t_assemble
        print '  Assemble into MatrixView', t_assemble_matview

        A1 -= A
        errornorm = A1.norm('linf')
        self.assertAlmostEqual(errornorm, 0.0)


        # (Deterministically) shuffle just owned dofs because we don't
        # know how much dofs including ghosts is there exactly
        ind = np.arange(num_dofs, dtype='uintp')
        range = V.dofmap().ownership_range()
        num_owned_dofs = range[1] - range[0]
        random.seed(42)
        random.shuffle(ind[:num_owned_dofs])

        # Zero actual storage; MatrixView.zero not implemented yet

        # Assemble into permuted matrix and compare norms
        # TODO: Compare also action to vector
        # NOTE: MatrixView.zero,norm not yet implemented, so calling to A1
        A1.zero()
        assemble(a, tensor=B, add_values=True)
        self.assertAlmostEqual(A.norm('l1'), A1.norm('l1'))


class VectorViewTest(unittest.TestCase):

    def test_full_view(self):

        x = Vector()
        AssemblerBase().init_global_tensor(x, Form(L))

        tic()
        assemble(L, tensor=x)
        t_assemble = toc()

        # Upper bound on dof count (including ghosts)
        num_dofs = V.dofmap().max_cell_dimension()*mesh.num_cells()
        ind = np.arange(num_dofs, dtype='uintp')

        x1 = x.copy()
        x1.zero()

        tic()
        y = VectorView(x1, V.dim(), ind)
        del ind # Check that ind is not garbage collected prematuraly
        t_vecview_constructor = toc()
        self.assertLess(t_vecview_constructor, 0.5)

        tic()
        assemble(L, tensor=y, add_values=True)
        t_assemble_vecview = toc()
        self.assertLess(t_assemble_vecview, 2.0*t_assemble)

        print 'Timings:'
        print '  Regular assemble        ', t_assemble
        print '  Assemble into VectorView', t_assemble_vecview

        x1 -= x
        errornorm = x1.norm('linf')
        self.assertAlmostEqual(errornorm, 0.0)


if __name__ == "__main__":
    unittest.main()
