from dolfin import *
import numpy as np
from petsc4py import PETSc

from common import MatrixView, VectorView, assemble
from common.utils import la_index_mpitype, num_nonzeros

__all__ = ['FluxReconstructor']

class FluxReconstructor(object):

    def __init__(self, mesh, degree):

        self._mesh = mesh

        # Color mesh to pathes
        vertex_colors, color_num = self._color_patches(mesh)
        self._color_num = color_num

        # Build dof mapping to patches problem
        self._build_dof_mapping(degree, vertex_colors)

        self._init_tensors()
        self._assemble_matrix()
        self._compress_matrix()
        self._setup_solver()
        self._build_hat_functions(vertex_colors)


    @staticmethod
    def _color_patches(mesh):
        parameters['graph_coloring_library'] = 'Boost'
        #parameters['graph_coloring_library'] = 'Zoltan'

        comm = mesh.mpi_comm()

        vertex_colors = VertexFunction('size_t', mesh)
        # TODO: These should give same result; which is cheaper?
        vertex_colors.array()[:] = MeshColoring.color(mesh, np.array([0, 1, 0], dtype='uintp'))
        #vertex_colors.array()[:] = MeshColoring.color(mesh, np.array([0, mesh.topology().dim(), 0], dtype='uintp'))
        color_num_local = int(vertex_colors.array().ptp()) + 1
        #TODO: Is it really needed? Optimally remove it
        #TODO: Switch to proper MPI int function
        color_num = int(MPI.max(comm, color_num_local))
        assert color_num >= color_num_local if MPI.size(comm) > 1 else color_num == color_num_local

        #max_uintp = int(np.uintp(-1))
        #cell_partitions = [CellFunction('size_t', mesh, max_uintp) for _ in xrange(color_num)]
        #for v in vertices(mesh):
        #    p = vertex_colors[v]
        #    for c in cells(v):
        #        cell_partitions[p][c] = p

        return vertex_colors, color_num


    def _assemble_matrix(self):
        u, r = TrialFunctions(self._W)
        v, q = TestFunctions(self._W)

        # TODO: restrict dx to patches
        a = ( inner(u, v) - inner(r, div(v)) - inner(q, div(u)) )*dx

        [assemble(a, tensor=self._A_patches[p], add_values=True,
                  finalize_tensor=False) for p in range(self._color_num)]
        self._A.apply('add')


    def _compress_matrix(self):
        # TODO: Rather do better preallocation than compression
        info_blue('Num nonzeros before compression: %d'%self._A.nnz())
        tic()
        A = PETScMatrix()
        self._A.compressed(A)
        self._A = A
        info_blue('Compression time: %g'%toc())
        info_blue('Num nonzeros after compression: %d'%self._A.nnz())


    def _setup_solver(self):
        # Diagnostic output
        #PETScOptions.set('mat_mumps_icntl_4', 3)
        #PETScOptions.set('mat_mumps_icntl_2', 6)

        self._solver = solver = PETScLUSolver('mumps')
        solver.set_operator(self._A)
        # https://bitbucket.org/petsc/petsc/issue/81
        #solver.parameters['symmetric'] = True
        solver.parameters['reuse_factorization'] = True


    def _build_hat_functions(self, vertex_colors):
        """Builds a list of hat functions for every partition."""
        # Define space, function and vector
        P1 = FunctionSpace(self._mesh, 'CG', 1)
        self._hat = hat = [Function(P1) for _ in range(self._color_num)]
        hat_vec = [u.vector() for u in hat]

        # Build hat DOFs
        hat_dofs = [np.zeros(x.local_size()) for x in hat_vec]
        dof_to_vertex = dof_to_vertex_map(P1)
        for dof, vertex in enumerate(dof_to_vertex):
            hat_dofs[vertex_colors[int(vertex)]][dof] = 1.0

        # Assign
        for i, x in enumerate(hat_vec):
            x[:] = hat_dofs[i]
        assert all(near(u.vector().max(), 1.0) for u in hat)


    def _init_tensors(self):
        comm = self._mesh.mpi_comm()
        color_num = self._color_num
        patches_dim_global = self._patches_dim_global
        patches_dim_offset = self._patches_dim_offset
        patches_dim_owned = self._patches_dim_owned
        dof_mapping = self._dof_mapping

        # Prepare tensor layout and buld sparsity pattern
        tl_A = TensorLayout(comm,
                            np.array(2*(patches_dim_global,), dtype='uintp'),
                            0,
                            1,
                            np.array(2*((patches_dim_offset, patches_dim_offset + patches_dim_owned), ), dtype='uintp'),
                            True)
        tl_b = TensorLayout(comm,
                            np.array((patches_dim_global,), dtype='uintp'),
                            0,
                            1,
                            np.array(((patches_dim_offset, patches_dim_offset + patches_dim_owned), ), dtype='uintp'),
                            False)
        self._build_sparsity_pattern(tl_A.sparsity_pattern())

        # Init tensors
        self._A = A = Matrix()
        self._b = b = Vector()
        A.init(tl_A)
        b.init(tl_b)
        self._x = x = b.copy()

        # Options to deal with zeros
        # TODO: Sort these options out
        as_backend_type(A).mat().setOption(
            PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)
        #as_backend_type(A).mat().setOption(
        #    PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)
        #as_backend_type(A).mat().setOption(
        #    PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR, True)
        #as_backend_type(A).mat().setOption(
        #    PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        # Enforce dropping of negative indices on VecSetValues
        as_backend_type(b).vec().setOption(
            PETSc.Vec.Option.IGNORE_NEGATIVE_INDICES, True)

        # Initialize tensor views to partitions
        self._A_patches = [MatrixView(A, self._W.dim(), self._W.dim(),
                                      dof_mapping[p], dof_mapping[p])
                           for p in range(color_num)]
        self._b_patches = [VectorView(b, self._W.dim(), dof_mapping[p])
                           for p in range(color_num)]
        self._x_patches = [VectorView(x, self._W.dim(), dof_mapping[p])
                           for p in range(color_num)]


    def _build_sparsity_pattern(self, pattern):
        mesh = self._mesh
        comm = self._mesh.mpi_comm()
        W = self._W
        color_num = self._color_num
        dof_mapping = self._dof_mapping
        patches_dim_offset = self._patches_dim_offset
        patches_dim_owned = self._patches_dim_owned
        patches_dim_global = self._patches_dim_global
        local_to_global_patches = self._local_to_global_patches
        off_process_owner = self._off_process_owner

        # TODO: Can we preallocate dict? Or do it otherwise better?
        global_to_local_patches_dict = {}
        for i, j in enumerate(local_to_global_patches):
            global_to_local_patches_dict[j] = i + patches_dim_owned
        assert MPI.size(comm) > 1 or len(global_to_local_patches_dict) == 0
        def global_to_local_patches(global_patch_dof):
            if patches_dim_offset <= global_patch_dof < patches_dim_offset + patches_dim_owned:
                return global_patch_dof - patches_dim_offset
            else:
                return global_to_local_patches_dict[global_patch_dof]

        pattern.init(comm,
            np.array(2*(patches_dim_global,), dtype='uintp'),
            np.array(2*((patches_dim_offset, patches_dim_offset + patches_dim_owned), ), dtype='uintp'),
            [local_to_global_patches, local_to_global_patches],
            [off_process_owner, []],
            np.array((1, 1), dtype='uintp'))

        # Build sparsity pattern for mixed Laplacian
        for c in cells(mesh):
            RT_dofs = W.sub(0).dofmap().cell_dofs(c.index())
            DG_dofs = W.sub(1).dofmap().cell_dofs(c.index())
            for p in range(color_num):
                RT_dofs_patch = dof_mapping[p][RT_dofs]
                DG_dofs_patch = dof_mapping[p][DG_dofs]
                RT_dofs_patch = RT_dofs_patch[RT_dofs_patch != -1]
                DG_dofs_patch = DG_dofs_patch[DG_dofs_patch != -1]
                RT_dofs_patch_owned_mask = np.logical_and(RT_dofs_patch >= patches_dim_offset,
                                                          RT_dofs_patch <  patches_dim_offset + patches_dim_owned)
                assert MPI.size(comm) > 1 or RT_dofs_patch_owned_mask.all()
                RT_dofs_patch_unowned_mask = np.logical_not(RT_dofs_patch_owned_mask)
                RT_dofs_patch_owned   = RT_dofs_patch[RT_dofs_patch_owned_mask]
                RT_dofs_patch_unowned = RT_dofs_patch[RT_dofs_patch_unowned_mask]
                pattern.insert_global([RT_dofs_patch_owned, RT_dofs_patch])
                pattern.insert_global([RT_dofs_patch_owned, DG_dofs_patch])
                pattern.insert_global([DG_dofs_patch,       RT_dofs_patch])
                if RT_dofs_patch_unowned.size > 0:
                    assert MPI.size(comm) > 1
                    DG_dofs_patch = map(global_to_local_patches, DG_dofs_patch)
                    RT_dofs_patch = map(global_to_local_patches, RT_dofs_patch)
                    RT_dofs_patch_unowned = map(global_to_local_patches, RT_dofs_patch_unowned)
                    pattern.insert_local([RT_dofs_patch_unowned, RT_dofs_patch])
                    pattern.insert_local([RT_dofs_patch_unowned, DG_dofs_patch])
        pattern.apply()


    def _build_dof_mapping(self, degree, vertex_colors):
        mesh = self._mesh
        color_num = self._color_num

        comm = mesh.mpi_comm()
        rank = MPI.rank(comm)
        size = MPI.size(comm)

        RT = FunctionSpace(mesh, 'Raviart-Thomas', degree+1)
        DG = FunctionSpace(mesh, 'Discontinuous Lagrange', degree)
        W = RT*DG
        self._W = W

        gdim = mesh.geometry().dim()
        tdim = mesh.topology().dim()
        num_cells = mesh.num_cells()
        num_facets = mesh.num_facets()
        num_vertices = mesh.num_vertices()
        dofs_per_vertex = W.dofmap().num_entity_dofs(0)
        dofs_per_facet = W.dofmap().num_entity_dofs(tdim-1)
        dofs_per_cell = W.dofmap().num_entity_dofs(tdim)

        assert dofs_per_vertex == 0

        mesh.init(0, tdim-1)
        mesh.init(0, tdim)
        def patch_dim(vertex):
            num_interior_facets = vertex.num_entities(tdim-1)
            num_patch_cells = vertex.num_entities(tdim)
            patch_dim = num_interior_facets*dofs_per_facet + num_patch_cells*dofs_per_cell
            # Remove one DG DOF per interior patch
            # TODO: this lowest rank distribution is not even!
            if not on_boundary(vertex) and \
              (not vertex.is_shared() or rank <= min(vertex.sharing_processes())):
                patch_dim -= 1
            return patch_dim

        mesh.init(tdim-1, tdim)
        def on_boundary(vertex):
            return any(f.exterior() for f in facets(vertex))

        num_dofs_with_ghosts = W.dofmap().local_dimension('all')
        patches_dim = tdim*dofs_per_facet*num_facets + (tdim + 1)*dofs_per_cell*num_cells
        # Remove one DG DOF per interior patch
        # TODO: this lowest rank distribution is not even!
        patches_dim -= sum(1 for v in vertices(mesh) if
                not on_boundary(v) and \
               (not v.is_shared() or rank <= min(v.sharing_processes())))
        partitions_dim = [sum(patch_dim(v) for v in vertices(mesh) if vertex_colors[v] == p)
                          for p in range(color_num)]

        assert sum(patch_dim(v) for v in vertices(mesh)) == patches_dim
        assert sum(partitions_dim) == patches_dim

        # Construct mapping of (rank-local) W dofs to (rank-global) patch-wise problems
        self._dof_mapping = dof_mapping = [np.empty(num_dofs_with_ghosts, dtype=la_index_dtype())
                       for p in range(color_num)]
        [dof_mapping[p].fill(-1) for p in range(color_num)]
        dof_counter = 0
        facet_dofs = [W.dofmap().tabulate_entity_dofs(tdim - 1, f)
                      for f in range(tdim + 1)]
        cell_dofs = W.dofmap().tabulate_entity_dofs(tdim, 0)
        local_ownership_size = W.dofmap().local_dimension('owned')
        local_to_global = W.dofmap().local_to_global_index
        shared_nodes = W.dofmap().shared_nodes()
        shared_dofs = {r: {} for r in sorted(W.dofmap().neighbours())}
        for v in vertices(mesh):
            p = vertex_colors[v]

            # Build local dofs
            local_dofs = []
            for c in cells(v):
                # TODO: Je to dobre? Je zde j spravny index facety?
                for j, f in enumerate(facets(c)):
                    # TODO: Don't call DofMap.cell_dofs redundantly
                    if f.incident(v): # Zero-flux on patch boundary
                        local_dofs += W.dofmap().cell_dofs(c.index())[facet_dofs[j]].tolist()
                local_dofs += W.dofmap().cell_dofs(c.index())[cell_dofs].tolist()
            # Remove one DG DOF per interior patch
            # TODO: this lowest rank distribution is not even!
            if not on_boundary(v) and \
              (not v.is_shared() or rank <= min(v.sharing_processes())):
                # TODO: This assertion is very costly!! Rework!
                assert local_dofs[-1] in W.sub(1).dofmap().dofs()-W.sub(1).dofmap().ownership_range()[0], \
                    (local_dofs, W.sub(1).dofmap().dofs()-W.sub(1).dofmap().ownership_range()[0])
                # TODO: Is it correct?! Isn't it be RT cell momentum?
                local_dofs = local_dofs[:-1]
            # Exclude unowned DOFs
            local_dofs = np.unique(dof for dof in local_dofs if dof < local_ownership_size)
            if local_dofs.size == 0:
                # TODO: Should this happen?!
                continue
            assert local_dofs.min() >= 0 and local_dofs.max() < local_ownership_size

            ## Build global dofs
            #num_dofs = patch_dim(v)
            #assert num_dofs == len(local_dofs)
            # Store to array; now with local indices
            num_dofs = len(local_dofs)
            assert num_dofs==patch_dim(v) if size==1 else num_dofs<=patch_dim(v)
            global_dofs = np.arange(dof_counter, dof_counter + num_dofs, dtype=la_index_dtype())

            # Store mapping and increase counter
            dof_mapping[p][local_dofs] = global_dofs
            dof_counter += num_dofs

            # Prepare shared dofs
            global_vertex_index = None
            for dof in local_dofs:
                # TODO: Check that dof is facet dof
                sharing_processes = shared_nodes.get(dof)
                if sharing_processes is None:
                    continue
                if global_vertex_index is None:
                    global_vertex_index = v.global_index()
                assert sharing_processes.size == 1
                global_dof = local_to_global(dof)
                r = sharing_processes[0]
                l = shared_dofs[r].get(global_dof)
                if l is None:
                    l = shared_dofs[r][global_dof] = []
                l += [global_vertex_index, dof_mapping[p][dof]]

        #assert dof_counter == patches_dim + patches_dim_offset
        assert dof_counter==patches_dim if size==1 else dof_counter<=patches_dim

        self._patches_dim_owned = patches_dim_owned = dof_counter
        self._patches_dim_offset = patches_dim_offset \
                = MPI.global_offset(comm, patches_dim_owned, True)
        for p in range(color_num):
            dof_mapping[p][dof_mapping[p]>=0] += patches_dim_offset # TODO: can we do it more clever?
        for d1 in shared_dofs.itervalues():
            for l in d1.itervalues():
                assert len(l) == 2*tdim
                for i in range(len(l)/2):
                    l[2*i+1] += patches_dim_offset

        from itertools import chain
        # TODO: Je zarucene poradi hodnot ve slovniku?!?
        c = chain.from_iterable(chain([v], data) for d in shared_dofs.itervalues()
                                                 for v, data in d.iteritems())
        sendbuf = np.fromiter(c, dtype=la_index_dtype())
        num_shared_unowned = W.dofmap().local_dimension('unowned')
        num_shared_owned = len(shared_nodes) - num_shared_unowned
        recvbuf = np.empty((2*tdim+1)*num_shared_unowned, dtype=la_index_dtype())
        sendcounts = np.zeros(size, dtype='intc')
        senddispls = np.zeros(size, dtype='intc')
        recvcounts = np.zeros(size, dtype='intc')
        recvdispls = np.zeros(size, dtype='intc')
        for dof, ranks in shared_nodes.iteritems():
            assert ranks.size == 1
            r = ranks[0]
            if dof < local_ownership_size:
                sendcounts[r]    += 2*tdim + 1
                senddispls[r+1:] += 2*tdim + 1
            else:
                recvcounts[r]    += 2*tdim + 1
                recvdispls[r+1:] += 2*tdim + 1

        assert sendcounts.sum() == sendbuf.size
        assert recvcounts.sum() == recvbuf.size

        # MPI_Alltoallv
        comm.tompi4py().Alltoallv((sendbuf, (sendcounts, senddispls), la_index_mpitype()),
                                  (recvbuf, (recvcounts, recvdispls), la_index_mpitype()))

        # Maps from unowned local patch dof to owning rank; Rp: ip' -> r
        self._off_process_owner = off_process_owner \
                = np.empty(patches_dim-patches_dim_owned, dtype='intc')
        # Maps from unowned local patch dof to global patch dof; Cp': ip' -> Ip
        self._local_to_global_patches = local_to_global_patches \
                = np.empty(patches_dim-patches_dim_owned, dtype=la_index_dtype())

        # Add unowned DOFs to dof_mapping
        dof_counter = 0
        mesh.init(tdim-1, tdim)
        for c in cells(mesh):
            c_dofs =  W.dofmap().cell_dofs(c.index())
            # TODO: Is this indexing correct?
            for i, f in enumerate(facets(c)):
                if f.num_entities(tdim) == 2 or f.exterior():
                    continue
                f_dofs = [dof for dof in c_dofs[facet_dofs[i]]
                              if dof>=local_ownership_size]
                if len(f_dofs) == 0:
                    continue
                for dof in f_dofs:
                    global_index = local_to_global(dof)
                    # TODO: restrict search to relevant process
                    # TODO: optimize this!
                    indices = recvbuf[::2*tdim+1]
                    assert indices.size*tdim == patches_dim-patches_dim_owned
                    # TODO: THIS MAY YIELD QUADRATIC SCALING !!!!!!!
                    j = np.where(indices==global_index)
                    assert len(j)==1 and j[0].size==1
                    j = (2*tdim + 1) * j[0][0]
                    assert recvbuf[j]==global_index
                    vertex_indices = recvbuf[j+1:j+1+2*tdim:2]
                    patch_dofs     = recvbuf[j+2:j+2+2*tdim:2]
                    # TODO: Maybe use DofMap::off_process_owner function
                    ranks = shared_nodes[dof]
                    assert ranks.size == 1
                    r = ranks[0]

                    for v in vertices(f):
                        p = vertex_colors[v]
                        k = np.where(vertex_indices==v.global_index())
                        assert len(k)==1 and k[0].size==1
                        k = k[0][0]
                        dof_mapping[p][dof] = patch_dofs[k]

                        off_process_owner[dof_counter] = r
                        local_to_global_patches[dof_counter] = patch_dofs[k]
                        dof_counter += 1

        assert dof_counter == patches_dim - patches_dim_owned
        #TODO: Consider sorting (local_to_global_patches, off_process_owner)

        # TODO: validity of dof_mapping needs to be tested!

        self._patches_dim_global = patches_dim_global = MPI.sum(comm, patches_dim_owned)


    def reconstruct(self, u, f):
        """Takes u, f and reconstructs H(div) gradient flux. Returns mixed
        function (q, r) where q approximates -grad(u) on RT space."""
        W = self._W
        b = self._b
        x = self._x
        b_patches = self._b_patches
        x_patches = self._x_patches
        color_num = self._color_num
        hat = self._hat
        solver = self._solver

        v, q = TestFunctions(W)
        b.zero()

        [assemble( (-hat[p]*inner(grad(u), v)
                    -hat[p]*f*q
                    +inner(grad(hat[p]), grad(u))*q)*dx,
                  tensor=b_patches[p], add_values=True)
          for p in range(color_num)]
        as_backend_type(b).update_ghost_values()

        solver.solve(x, b)

        w = Function(W)
        [xp.add_to_vector(w.vector()) for xp in x_patches]

        return w


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import ufl
    ufl.set_level(ufl.INFO) # Enable info_{green,red,blue}

    results = ([], [], [], [], [])

    for N in [2**i for i in range(2, 7)]:
        mesh = UnitSquareMesh(N, N)
        V = FunctionSpace(mesh, 'CG', 1)
        u, v = TrialFunction(V), TestFunction(V)
        a = inner(grad(u), grad(v))*dx
        m, n = 1, 1
        u_ex = Expression('sin(m*pi*x[0])*sin(n*pi*x[1])', m=m, n=n, degree=4)
        f = Constant((m*m + n*n)*pi*pi)*u_ex
        L = f*v*dx
        bc = DirichletBC(V, 0.0, lambda x, b: b)
        u = Function(V)
        solve(a == L, u, bc)

        reconstructor = FluxReconstructor(mesh, 2) # TODO: Is it correct degree?
        w = reconstructor.reconstruct(u, f)
        q = Function(w, 0)

        plot(q)

        # TODO: h is not generally constant!
        h = sqrt(2.0)/N
        results[0].append((N, V.dim()))
        energy_error = errornorm(u_ex, u, norm_type='H10')

        # Approximation; works surprisingly well
        err_est = assemble(inner(grad(u)+q, grad(u)+q)*dx)**0.5 \
                + h/pi*assemble(inner(f-div(q), f-div(q))*dx)**0.5
        info_red('Estimator %g, energy_error %g' % (err_est, energy_error))
        results[1].append((err_est, energy_error))

        # Correct way; slow numpy manipulation is used
        DG0 = FunctionSpace(mesh, 'DG', 0)
        v = TestFunction(DG0)
        err0 = assemble(inner(grad(u)+q, grad(u)+q)*v*dx)
        err1 = assemble(inner(f-div(q), f-div(q))*v*dx)
        err0[:] = err0.array()**0.5 + h/pi*err1.array()**0.5
        err_est = err0.norm('l2')
        info_red('Estimator %g, energy_error %g' % (err_est, energy_error))
        results[2].append((err_est, energy_error))

        # Approximation
        err_est = assemble ( ( inner(grad(u)+q, grad(u)+q)**0.5
                             + Constant(h/pi)*inner(f-div(q), f-div(q))**0.5
                             )**2*dx
                           ) ** 0.5
        info_red('Estimator %g, energy_error %g' % (err_est, energy_error))
        results[3].append((err_est, energy_error))

        # Other ways of computing error
        u_err = errornorm(u_ex, u)
        q_ex = Expression(('-m*pi*cos(m*pi*x[0])*sin(n*pi*x[1])',
                           '-n*pi*sin(m*pi*x[0])*cos(n*pi*x[1])'),
                           m=m, n=n, degree=3)
        q_err = errornorm(q_ex, q)
        info_red('u L2-errornorm %g, q L2-errornorm %g)'%(u_err, q_err))
        results[4].append((0.0, q_err))
        Q = w.function_space().sub(0).collapse()
        info_red('||grad(u)+q||_2 = %g'%norm(project(grad(u)+q, Q)))
        info_red('||grad(u)-grad(u_ex)||_2 = %g'%errornorm(q_ex, project(-grad(u), Q)))

    results = np.array(results, dtype='float')

    plt.subplot(4, 1, 1)
    plt.plot(results[0, :, 0], results[1, :, 0], 'o-')
    plt.plot(results[0, :, 0], results[1, :, 1], 'o-')
    plt.loglog()
    plt.subplot(4, 1, 2)
    plt.plot(results[0, :, 0], results[2, :, 0], 'o-')
    plt.plot(results[0, :, 0], results[2, :, 1], 'o-')
    plt.loglog()
    plt.subplot(4, 1, 3)
    plt.plot(results[0, :, 0], results[3, :, 0], 'o-')
    plt.plot(results[0, :, 0], results[3, :, 1], 'o-')
    plt.loglog()
    plt.subplot(4, 1, 4)
    plt.plot(results[0, :, 0], results[4, :, 0], 'o-')
    plt.plot(results[0, :, 0], results[4, :, 1], 'o-')
    plt.loglog()

    plt.tight_layout()
    plt.show(block=True)

    interactive()
