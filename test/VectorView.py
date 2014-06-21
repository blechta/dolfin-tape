import unittest
from dolfin import *
import numpy as np

header_file = open("../common/VectorView.h")
code = "\n".join(header_file.readlines())
header_file.close()
VectorView_module = compile_extension_module(code)
VectorView = VectorView_module.VectorView

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

        # NOTE: User must take care of ind not being garbage
        #       collected during lifetime of y!
        ind = np.arange(V.dim(), dtype='uintp')

        x1 = x.copy()
        x1.zero()

        tic()
        y = VectorView(x1, ind)
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
