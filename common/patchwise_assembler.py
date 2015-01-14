from dolfin import *
import numpy as np
from petsc4py import PETSc

from common import MatrixView, VectorView, assemble
from common.utils import la_index_mpitype, num_nonzeros

__all__ = ['FluxReconstructor']

parameters['graph_coloring_library'] = 'Boost'
#parameters['graph_coloring_library'] = 'Zoltan'

class FluxReconstructor(object):

    def __init__(self, mesh, degree):

        self._mesh = mesh

        comm = mesh.mpi_comm()
        rank = MPI.rank(comm)
        size = MPI.size(comm)

        vertex_colors = VertexFunction('size_t', mesh)
        # TODO: These should give same result; which is cheaper?
        vertex_colors.array()[:] = MeshColoring.color(mesh, np.array([0, 1, 0], dtype='uintp'))
        #vertex_colors.array()[:] = MeshColoring.color(mesh, np.array([0, mesh.topology().dim(), 0], dtype='uintp'))
        color_num_local = int(vertex_colors.array().ptp()) + 1
        #TODO: Is it really needed? Optimally remove it
        #TODO: Switch to proper MPI int function
        color_num = int(MPI.max(comm, color_num_local))
        assert color_num >= color_num_local if size > 1 else color_num == color_num_local
        self._color_num = color_num

        max_uintp = int(np.uintp(-1))
        cell_partitions = [CellFunction('size_t', mesh, max_uintp) for _ in xrange(color_num)]
        for v in vertices(mesh):
            p = vertex_colors[v]
            for c in cells(v):
                cell_partitions[p][c] = p

        #plot(vertex_colors)
        #for cp in cell_partitions:
        #    plot(cp)
        #interactive()

        l = degree
        RT = FunctionSpace(mesh, 'Raviart-Thomas', l+1)
        DG = FunctionSpace(mesh, 'Discontinuous Lagrange', l)
        W = RT*DG
        self._W = W

        u, r = TrialFunctions(W)
        v, q = TestFunctions(W)

        # TODO: restrict dx to patches
        a = ( inner(u, v) - inner(r, div(v)) - inner(q, div(u)) )*dx

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

        num_dofs_with_ghosts = W.dofmap().tabulate_local_to_global_dofs().size # TODO: pull request #175
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
        dof_mapping = [np.empty(num_dofs_with_ghosts, dtype=la_index_dtype())
                       for p in range(color_num)]
        [dof_mapping[p].fill(-1) for p in range(color_num)]
        dof_counter = 0
        facet_dofs = [W.dofmap().tabulate_entity_dofs(tdim - 1, f)
                      for f in range(tdim + 1)]
        cell_dofs = W.dofmap().tabulate_entity_dofs(tdim, 0)
        local_ownership_size = W.dofmap().ownership_range()[1] - W.dofmap().ownership_range()[0]
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

        patches_dim_owned = dof_counter
        patches_dim_offset = MPI.global_offset(comm, patches_dim_owned, True)
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
        num_shared_unowned = - W.dofmap().ownership_range()[1] \
                             + W.dofmap().ownership_range()[0] \
                             + W.dofmap().tabulate_local_to_global_dofs().size # TODO: Avoid this costly function!
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
        off_process_owner = np.empty(patches_dim-patches_dim_owned, dtype='intc')
        # Maps from unowned local patch dof to global patch dof; Cp': ip' -> Ip
        local_to_global_patches = np.empty(patches_dim-patches_dim_owned, dtype=la_index_dtype())

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

        A = Matrix()
        b = Vector()

        patches_dim_global = MPI.sum(comm, patches_dim_owned)

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

        sp_A = tl_A.sparsity_pattern()
        sp_A.init(comm,
                  np.array(2*(patches_dim_global,), dtype='uintp'),
                  np.array(2*((patches_dim_offset, patches_dim_offset + patches_dim_owned), ), dtype='uintp'),
                  [local_to_global_patches, local_to_global_patches],
                  [off_process_owner, []],
                  np.array((1, 1), dtype='uintp'))

        # TODO: Can we preallocate dict? Or do it otherwise better?
        global_to_local_patches_dict = {}
        for i, j in enumerate(local_to_global_patches):
            global_to_local_patches_dict[j] = i + patches_dim_owned
        assert size > 1 or len(global_to_local_patches_dict) == 0
        def global_to_local_patches(global_patch_dof):
            if patches_dim_offset <= global_patch_dof < patches_dim_offset + patches_dim_owned:
                return global_patch_dof - patches_dim_offset
            else:
                return global_to_local_patches_dict[global_patch_dof]

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
                assert size > 1 or RT_dofs_patch_owned_mask.all()
                RT_dofs_patch_unowned_mask = np.logical_not(RT_dofs_patch_owned_mask)
                RT_dofs_patch_owned   = RT_dofs_patch[RT_dofs_patch_owned_mask]
                RT_dofs_patch_unowned = RT_dofs_patch[RT_dofs_patch_unowned_mask]
                sp_A.insert_global([RT_dofs_patch_owned, RT_dofs_patch])
                sp_A.insert_global([RT_dofs_patch_owned, DG_dofs_patch])
                sp_A.insert_global([DG_dofs_patch,       RT_dofs_patch])
                sp_A.insert_global([DG_dofs_patch,       DG_dofs_patch]) # TODO: Remove this!
                if RT_dofs_patch_unowned.size > 0:
                    assert size > 1
                    DG_dofs_patch = map(global_to_local_patches, DG_dofs_patch)
                    RT_dofs_patch = map(global_to_local_patches, RT_dofs_patch)
                    RT_dofs_patch_unowned = map(global_to_local_patches, RT_dofs_patch_unowned)
                    sp_A.insert_local([RT_dofs_patch_unowned, RT_dofs_patch])
                    sp_A.insert_local([RT_dofs_patch_unowned, DG_dofs_patch])
        sp_A.apply()

        b.init(tl_b)
        A.init(tl_A)
        x = b.copy()

        #PETScOptions.set('log_summary')
        #print parameters['use_petsc_signal_handler'], parameters['foo']

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

        A_patches = [MatrixView(A, W.dim(), W.dim(), dof_mapping[p], dof_mapping[p]) for p in range(color_num)]
        [assemble(a, tensor=A_patches[p], add_values=True, finalize_tensor=False) for p in range(color_num)]
        A.apply('add')

        print A.norm('frobenius'), A.norm('l1'), A.norm('linf')
        if size == 1:
            print as_backend_type(A).mat().isSymmetric(), as_backend_type(A).mat().isSymmetric(1e-6)
        #print as_backend_type(A).mat().isStructurallySymmetric()
        #print as_backend_type(A).mat().isSymmetricKnown()
        #arr = A.array()
        #ApAT = arr-arr.T
        #print sum(sum(abs(ApAT))), ApAT.min(), ApAT.max()

        print 'mat size', A.size(0), A.size(1)
        print 'num cells*4', 4*mesh.num_cells() - mesh.num_vertices()
        print 'num nonzeros', num_nonzeros(as_backend_type(A))
        tic()
        #Ac = PETScMatrix()
        #A.compressed(Ac)
        #A = Ac # Garbage collection
        print 'compression', toc()
        print 'num nonzeros', num_nonzeros(A)

        print A.norm('frobenius'), A.norm('l1'), A.norm('linf')
        if size == 1:
            print as_backend_type(A).mat().isSymmetric(), as_backend_type(A).mat().isSymmetric(1e-6)
        #exit()


        #b = Vector()
        #x = Vector()
        #A.init_vector(b, 0)
        #A.init_vector(x, 1)

        self._b = b
        self._x = x

        b_patches = [VectorView(b, W.dim(), dof_mapping[p]) for p in range(color_num)]
        x_patches = [VectorView(x, W.dim(), dof_mapping[p]) for p in range(color_num)]
        #[assemble(TestFunctions(W)[1]*dx, tensor=b_patches[p], add_values=True) for p in range(color_num)]
        [assemble(TestFunctions(W)[1]*dx, tensor=b_patches[p], add_values=True) for p in [0]]
        as_backend_type(b).update_ghost_values()

        self._b_patches = b_patches
        self._x_patches = x_patches

        #==========================================================================

        #methods = ['mumps', 'petsc', 'umfpack', 'superlu', 'superlu_dist']
        #methods = ['mumps', 'umfpack', 'superlu', 'superlu_dist']
        methods = ['mumps']
        #PETScOptions.set('mat_mumps_icntl_4', 3)
        #PETScOptions.set('mat_mumps_icntl_2', 6)
        for method in methods:
            solver = PETScLUSolver(method)
            solver.set_operator(A)
            #solver.parameters['verbose'] = True
            # TODO: Problem je, ze PETSc MatConvertToTriples_seqaij_seqsbaij predpoklada, ze matice je
            #       strukturalne symetricka pri alokaci, takze muze naalokovat spatnou delku vstupu pro MUMPS
            #solver.parameters['symmetric'] = True
            solver.parameters['reuse_factorization'] = True
            tic()
            solver.solve(x, b)
            print method, toc()
            tic()
            solver.solve(x, b)
            print method, toc()

        self._solver = solver

        P1 = FunctionSpace(mesh, 'CG', 1)
        hat = [Function(P1) for _ in range(color_num)]
        hat_vec = [u.vector() for u in hat]
        dof_to_vertex = dof_to_vertex_map(P1)
        for dof, vertex in enumerate(dof_to_vertex):
            hat_vec[vertex_colors[int(vertex)]][dof] = 1.0
        assert all(near(u.vector().max(), 1.0) for u in hat)
        #for u in hat:
        #    plot(u)
        #interactive()
        self._hat = hat


    def reconstruct(self, u, f):

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
    import ufl
    ufl.set_level(ufl.INFO) # Enable info_{green,red,blue}

    for N in [2**i for i in range(7)]:
        mesh = UnitSquareMesh(N, N)
        V = FunctionSpace(mesh, 'CG', 1)
        u, v = TrialFunction(V), TestFunction(V)
        a = inner(grad(u), grad(v))*dx
        m, n = 1, 1
        u_ex = Expression('sin(m*pi*x[0])*sin(n*pi*x[1])', m=m, n=n, degree=2)
        f = Constant((m*m + n*n)*pi*pi)*u_ex
        L = f*v*dx
        bc = DirichletBC(V, 0.0, lambda x, b: b)
        u = Function(V)
        solve(a == L, u, bc)

        reconstructor = FluxReconstructor(mesh, 1) # TODO: Is it correct degree?
        w = reconstructor.reconstruct(u, f)
        q = Function(w, 0)

        u_err = errornorm(u_ex, u)
        q_ex = Expression(('-m*pi*cos(m*pi*x[0])*sin(n*pi*x[1])',
                           '-n*pi*sin(m*pi*x[0])*cos(n*pi*x[1])'),
                           m=m, n=n, degree=2)
        q_err = errornorm(q_ex, q)

        info_red('u l2-errornorm %g, q l2-errornorm %g)'%(u_err, q_err))
        plot(q)

    interactive()
