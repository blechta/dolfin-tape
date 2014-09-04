import unittest
from dolfin import *
from petsc4py import PETSc
import numpy as np
import random

from common import MatrixView, VectorView

class BaseCase(unittest.TestCase):

    def setUp(self):

        mesh = UnitSquareMesh(*self.nx)
        V = FunctionSpace(mesh, 'CG', 1)
        u, v = TrialFunction(V), TestFunction(V)
        self.a = inner(grad(u), grad(v))*dx
        self.L = v*dx

        # Global number DOFs
        self.dim = V.dim()

        # Number DOFs including ghosts; Lagrange deg 1 assumend
        self.num_dofs = mesh.num_vertices()

        # Number owned DOFs
        range = V.dofmap().ownership_range()
        self.num_owned_dofs = range[1] - range[0]

        # Indices providing Matrix/VectorView mapping
        self.ind = np.arange(self.num_dofs, dtype='uintp')


class LargeCase(BaseCase):

    nx = (2000, 300)

    def test_matrix_view(self):

        A = Matrix()
        AssemblerBase().init_global_tensor(A, Form(self.a))

        tic()
        assemble(self.a, tensor=A)
        t_assemble = toc()

        A1 = A.copy()
        A1.zero()

        tic()
        B = MatrixView(A1, self.dim, self.dim, self.ind, self.ind)
        t_matview_constructor = toc()
        self.assertLess(t_matview_constructor, 0.5)

        tic()
        assemble(self.a, tensor=B, add_values=True)
        t_assemble_matview = toc()
        self.assertLess(t_assemble_matview, 2.0*t_assemble)

        print 'Timings:'
        print '  Regular assemble                 ', t_assemble
        print '  Assemble into MatrixView         ', t_assemble_matview

        A1 -= A
        errornorm = A1.norm('linf')
        self.assertAlmostEqual(errornorm, 0.0)


    def test_vector_view(self):

        x = Vector()
        AssemblerBase().init_global_tensor(x, Form(self.L))

        tic()
        assemble(self.L, tensor=x)
        t_assemble = toc()

        x1 = x.copy()
        x1.zero()

        tic()
        y = VectorView(x1, self.dim, self.ind)
        t_vecview_constructor = toc()
        self.assertLess(t_vecview_constructor, 0.5)

        tic()
        assemble(self.L, tensor=y, add_values=True)
        t_assemble_vecview = toc()
        self.assertLess(t_assemble_vecview, 2.0*t_assemble)

        x1 -= x
        errornorm = x1.norm('linf')
        self.assertAlmostEqual(errornorm, 0.0)

        # Shuffle DOFs; owned and ghosts separately
        random.seed(42)
        random.shuffle(self.ind[:self.num_owned_dofs ])
        random.shuffle(self.ind[ self.num_owned_dofs:])

        x1.zero() # NOTE: VectorView.zero,norm not yet implemented, so calling to x1
        tic()
        assemble(self.L, tensor=y, add_values=True)
        t_assemble_vecview_shuffled = toc()

        # Compare norms
        self.assertAlmostEqual(x.norm('l1'), x1.norm('l1'))

        # Check that arrays are the same
        # NOTE: Works only sequentially as we did not shuffle ghosts properly
        if MPI.size(mpi_comm_world()) == 1:
            self.assertAlmostEqual(
                    sum(abs(x1.array()[self.ind[:self.num_owned_dofs]]-x.array())),
                    0.0)

        print 'Timings:'
        print '  Regular assemble                 ', t_assemble
        print '  Assemble into VectorView         ', t_assemble_vecview
        print '  Assemble into shuffled VectorView', t_assemble_vecview_shuffled


class SmallCase(BaseCase):

    nx = (20, 30)

    def test_shuffled_matrix_view(self):

        A = assemble(self.a)
        A1 = A.copy()
        A1.zero()
        as_backend_type(A1).mat().setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        B = MatrixView(A1, self.dim, self.dim, self.ind, self.ind)

        assemble(self.a, tensor=B, add_values=True)


if __name__ == "__main__":
    unittest.main()
