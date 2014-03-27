import unittest
from dolfin import *
import numpy as np

header_file = open("../common/MatrixView.h")
code = "\n".join(header_file.readlines())
header_file.close()
MatrixView_module = compile_extension_module(code)
MatrixView = MatrixView_module.MatrixView

class MatrixViewTest(unittest.TestCase):

    def test_full_view(self):

        m, n = 2000, 300
        mesh = UnitSquareMesh(m, n)
        V = FunctionSpace(mesh, 'CG', 1)
        u, v = TrialFunction(V), TestFunction(V)
        a = inner(grad(u), grad(v))*dx
 
        A = Matrix()
        AssemblerBase().init_global_tensor(A, Form(a))

        tic()
        assemble(a, tensor=A)
        t_assemble = toc()

        # NOTE: User must take care of ind not being garbage
        #       collected during lifetime of B!
        ind = np.arange(V.dim(), dtype='uintp')

        A1 = A.copy()
        A1.zero()
        
        tic()
        B = MatrixView(A1, ind, ind)
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

if __name__ == "__main__":
    unittest.main()
