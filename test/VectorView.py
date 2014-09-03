import unittest
from dolfin import *
import numpy as np

from common import VectorView

class VectorViewTest(unittest.TestCase):

    def test_full_view(self):

        m, n = 2000, 300
        mesh = UnitSquareMesh(m, n)
        V = FunctionSpace(mesh, 'CG', 1)
        u, v = TrialFunction(V), TestFunction(V)
        L = v*dx

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
