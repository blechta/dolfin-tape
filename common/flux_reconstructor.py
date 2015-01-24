from dolfin import *
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI as MPI4py

from itertools import chain

from tensor_views import MatrixView, VectorView, assemble
from utils import la_index_mpitype

__all__ = ['FluxReconstructor']

class FluxReconstructor(object):

    def __init__(self, mesh, degree):
        """Build flux reconstructor of given degree."""
        self._mesh = mesh

        # Color mesh vertices
        color_num, vertex_colors, cell_partitions = self._color_patches(mesh)
        self._color_num = color_num
        self._cell_partitions = cell_partitions

        # Build dof mapping to patches problem
        self._build_dof_mapping(degree, vertex_colors)

        self._init_tensors()
        self._assemble_matrix()
        self._compress_matrix()
        self._setup_solver()
        self._build_hat_functions(vertex_colors)

        # TODO: Clear unneeded mesh connectivity
        # TODO: Delete unneeded tensor layouts and sparsity pattern


    def reconstruct(self, u, f):
        """Takes u, f and reconstructs H(div) gradient flux. Returns mixed
        function (q, r) where q approximates -grad(u) on RT space."""
        v, q = TestFunctions(self._W)
        hat = self._hat

        # Linear form on patches of color p
        def L(p):
            dxp = dx(subdomain_id=1, subdomain_data=self._cell_partitions[p])
            return ( -hat[p]*inner(grad(u), v)
                     -hat[p]*f*q
                     +inner(grad(hat[p]), grad(u))*q )*dxp

        # Assemble rhs color by color
        self._b.zero()
        for p in xrange(self._color_num):
            assemble(L(p), tensor=self._b_patches[p],
                     add_values=True, finalize_tensor=False)
        self._b.apply('add')

        # Solve the system
        assert self._solver, "Solver has not been initialized yet."
        self._solver.solve(self._x, self._b)

        # Reuse function for holding result
        try:
            w = self._w
        except AttributeError:
            w = self._w = Function(self._W)
        else:
            w.vector().zero()

        # Collect the result from patches to usual space
        for x_p in self._x_patches:
            x_p.add_to_vector(w.vector())
        w.vector().apply('add')

        return w


    @staticmethod
    def _color_patches(mesh):
        """Returns number of colors, vertex function with colors and sequence
        of cell functions cp s.t.:
            cp[p][c] = 1 ... if c belongs to patch of color p
            cp[p][c] = 0 ... otherwise
        """
        comm = mesh.mpi_comm()

        # TODO: Does Zoltan provide better coloring?
        parameters['graph_coloring_library'] = 'Boost'
        #parameters['graph_coloring_library'] = 'Zoltan'

        # Color vertices (equaivalent to coloring patches)
        # TODO: 0-1-0 and 0-d-0 should give same result; which is cheaper?
        coloring_type = np.array([0, 1, 0], dtype='uintp')
        #coloring_type = np.array([0, mesh.topology().dim(), 0], dtype='uintp')
        vertex_colors = VertexFunction('size_t', mesh)
        vertex_colors.array()[:] = MeshColoring.color(mesh, coloring_type)

        # Compute color number
        color_num_local = int(vertex_colors.array().ptp()) + 1
        sendbuf = np.array(color_num_local)
        recvbuf = np.array(0)
        comm.tompi4py().Allreduce(sendbuf, recvbuf, op=MPI4py.MAX)
        color_num = int(recvbuf)
        assert color_num >= color_num_local if MPI.size(comm) > 1 \
                else color_num == color_num_local

        # Build cell partitions
        # TODO: Cell partitions are used merely for restricting integration to
        #       patches. It is not obvious whether this overhead is worth.
        cell_partitions = [CellFunction('size_t', mesh)
                           for _ in xrange(color_num)]
        for v in vertices(mesh):
            p = vertex_colors[v]
            for c in cells(v):
                cell_partitions[p][c] = 1

        return color_num, vertex_colors, cell_partitions


    def _assemble_matrix(self):
        """Assembles flux reconstruction matrix on all the patches. Mixed
        Poisson like saddle-point form is used.

        The matrix is stored as self._A but accessed by assembler through
        matrix views self._A_patches[:]."""
        u, r = TrialFunctions(self._W)
        v, q = TestFunctions(self._W)

        # Bilinear form on patches of color p
        def a(p):
            dxp = dx(subdomain_id=1, subdomain_data=self._cell_partitions[p])
            return ( inner(u, v) - inner(r, div(v)) - inner(q, div(u)) )*dxp

        # Assemble matrix color by color
        # NOTE: assuming self._A is zeroed
        for p in xrange(self._color_num):
            assemble(a(p), tensor=self._A_patches[p],
                     add_values=True, finalize_tensor=False)
        self._A.apply('add')

        # Ensure that this method is not called twice (to avoid zeroing matrix)
        def on_second_call():
            raise RuntimeError("_assemble_matrix can't be called twice!")
        self._assemble_matrix = lambda *args, **kwargs: on_second_call()


    def _compress_matrix(self):
        """Removes (nearly) zero entries from self._A sparsity pattern."""
        # TODO: Rather do better preallocation than compression
        info_blue('Num nonzeros before compression: %d'%self._A.nnz())
        tic()
        A = PETScMatrix()
        self._A.compressed(A)
        self._A = A
        info_blue('Compression time: %g'%toc())
        info_blue('Num nonzeros after compression: %d'%self._A.nnz())


    def _setup_solver(self):
        """Initilize Cholesky solver for solving flux reconstruction."""
        # Diagnostic output
        #PETScOptions.set('mat_mumps_icntl_4', 3)
        #PETScOptions.set('mat_mumps_icntl_2', 6)

        self._solver = solver = PETScLUSolver('mumps')
        #self._solver = solver = PETScLUSolver('superlu_dist')
        solver.set_operator(self._A)
        # FIXME: Allow Cholesky only with fixed version of PETSc
        #        https://bitbucket.org/petsc/petsc/issue/81
        #        Wait for next PETSc release and check PETSc version.
        solver.parameters['symmetric'] = True
        solver.parameters['reuse_factorization'] = True


    def _build_hat_functions(self, vertex_colors):
        """Builds a list of hat functions for every partition."""
        #FIXME: We create one P1 function for every partition and rank as
        #       vertex coloring is not compatible across partition boundaries
        #       hence hat functions need to be discontinuous. This is handled
        #       by creating hat functions s.t. every single one takes hat
        #       values on just one parition/rank combination. This is hack
        #       (it may not be guaranteed that passing different cofficient
        #       to form is correct) but moreover it is not scalable solution!

        rank = MPI.rank(self._mesh.mpi_comm())
        size = MPI.size(self._mesh.mpi_comm())

        # Define space, function and vector
        P1 = FunctionSpace(self._mesh, 'CG', 1)
        hat = [Function(P1) for _ in range(size*self._color_num)]
        hat_vec = [u.vector() for u in hat]

        # Build hat DOFs
        hat_dofs = [[] for x in hat_vec]
        vertex_to_dof = vertex_to_dof_map(P1)
        for vertex, dof in enumerate(vertex_to_dof):
            hat_dofs[vertex_colors[int(vertex)]*size+rank].append(dof)
        hat_dofs = [np.array(hd, dtype=la_index_dtype()) for hd in hat_dofs]

        # Assign
        for i, x in enumerate(hat_vec):
            if i % size == rank:
                x.set_local(np.ones(hat_dofs[i].shape, dtype='float_'), hat_dofs[i])
            x.apply('insert')

        hat = [hat[p*size+rank] for p in range(self._color_num)]
        self._hat = hat


    def _init_tensors(self):
        comm = self._mesh.mpi_comm()
        color_num = self._color_num
        patches_dim_global = self._patches_dim_global
        patches_dim_offset = self._patches_dim_offset
        patches_dim_owned = self._patches_dim_owned
        patches_dim = self._patches_dim
        dof_mapping = self._dof_mapping
        _local_to_global_patches = self._local_to_global_patches

        # Prepare tensor layout and build sparsity pattern
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

        # Write local_to_global to tensor layouts
        # TODO: Is it needed?
        local_to_global_patches = np.arange(patches_dim_offset, patches_dim_offset + patches_dim, dtype='uintp')
        local_to_global_patches[patches_dim_owned:] = _local_to_global_patches
        tl_A.local_to_global_map[0] = local_to_global_patches
        tl_A.local_to_global_map[1] = local_to_global_patches
        tl_b.local_to_global_map[0] = local_to_global_patches

        self._build_sparsity_pattern(tl_A.sparsity_pattern())

        # Init matrix
        self._A = A = PETScMatrix()
        A.init(tl_A)

        ## Init vectors using tensor layout (without ghosts)
        #self._b = b = PETScVector()
        #b.init(tl_b)
        #self._x = x = b.copy()

        # Init vectors witht ghosts
        # TODO: Is it needed?
        self._b = b = PETScVector()
        self._x = x = PETScVector()
        b.init(comm, (patches_dim_offset, patches_dim_offset + patches_dim_owned),
               local_to_global_patches, _local_to_global_patches)
        x.init(comm, (patches_dim_offset, patches_dim_offset + patches_dim_owned),
               local_to_global_patches, _local_to_global_patches)

        # Options to deal with zeros
        # TODO: Sort these options out
        A.mat().setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)
        #A.mat().setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, False)
        #A.mat().setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR, True)
        #A.mat().setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        # Enforce dropping of negative indices on Vec(Get|Set)Values
        b.vec().setOption(PETSc.Vec.Option.IGNORE_NEGATIVE_INDICES, True)
        x.vec().setOption(PETSc.Vec.Option.IGNORE_NEGATIVE_INDICES, True)

        # Initialize tensor views to partitions
        self._A_patches = [MatrixView(A, self._W.dim(), self._W.dim(),
                                      dof_mapping[p], dof_mapping[p])
                           for p in xrange(color_num)]
        self._b_patches = [VectorView(b, self._W.dim(), dof_mapping[p])
                           for p in xrange(color_num)]
        self._x_patches = [VectorView(x, self._W.dim(), dof_mapping[p])
                           for p in xrange(color_num)]


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
            for p in xrange(color_num):
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
        del RT, DG
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
        ##def patch_dim(vertex):
        ##    num_interior_facets = vertex.num_entities(tdim-1)
        ##    num_patch_cells = vertex.num_entities(tdim)
        ##    patch_dim = num_interior_facets*dofs_per_facet + num_patch_cells*dofs_per_cell
        ##    # Remove one DG DOF per interior patch
        ##    # TODO: this lowest rank distribution is not even!
        ##    if not on_boundary(vertex) and \
        ##      (not vertex.is_shared() or rank <= min(vertex.sharing_processes())):
        ##        patch_dim -= 1
        ##    return patch_dim

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
        self._patches_dim = patches_dim
        ##partitions_dim = [sum(patch_dim(v) for v in vertices(mesh) if vertex_colors[v] == p)
        ##                  for p in xrange(color_num)]

        ##assert sum(patch_dim(v) for v in vertices(mesh)) == patches_dim
        ##assert sum(partitions_dim) == patches_dim

        # Construct mapping of (rank-local) W dofs to (rank-global) patch-wise problems
        self._dof_mapping = dof_mapping = [np.empty(num_dofs_with_ghosts, dtype=la_index_dtype())
                       for p in xrange(color_num)]
        [dof_mapping[p].fill(-1) for p in xrange(color_num)]
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
                #assert local_dofs[-1] in W.sub(1).dofmap().dofs()-W.sub(1).dofmap().ownership_range()[0], \
                #    (local_dofs, W.sub(1).dofmap().dofs()-W.sub(1).dofmap().ownership_range()[0])
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
            ##assert num_dofs==patch_dim(v) if size==1 else num_dofs<=patch_dim(v)
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
        for p in xrange(color_num):
            dof_mapping[p][dof_mapping[p]>=0] += patches_dim_offset # TODO: can we do it more clever?
        for d1 in shared_dofs.itervalues():
            for l in d1.itervalues():
                assert len(l) == 2*tdim
                for i in range(len(l)/2):
                    l[2*i+1] += patches_dim_offset

        c = chain.from_iterable(chain([v], data)
                                for r in sorted(shared_dofs.iterkeys())
                                for v, data in shared_dofs[r].iteritems())
        sendbuf = np.fromiter(c, dtype=la_index_dtype())
        num_shared_unowned = W.dofmap().local_dimension('unowned')
        num_shared_owned = len(shared_nodes) - num_shared_unowned
        assert sendbuf.size == (2*tdim+1)*num_shared_owned
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
                        assert 0 <= k < tdim
                        dof_mapping[p][dof] = patch_dofs[k]

                        off_process_owner[dof_counter] = r
                        local_to_global_patches[dof_counter] = patch_dofs[k]
                        dof_counter += 1

        assert dof_counter == patches_dim - patches_dim_owned

        #TODO: Consider sorting (local_to_global_patches, off_process_owner)
        #local_to_global_patches.sort()
        #off_process_owner.sort()

        # TODO: validity of dof_mapping needs to be tested!

        self._patches_dim_global = MPI.sum(comm, patches_dim_owned)

        # Debugging
        #temp = np.ndarray((2+color_num, num_dofs_with_ghosts))
        #local_dofs = np.arange(num_dofs_with_ghosts)
        #global_dofs = map(local_to_global, local_dofs)
        #temp[0] = local_dofs
        #temp[1] = global_dofs
        #for p in xrange(color_num):
        #    patch_dofs = map(lambda dof: dof_mapping[p][dof], local_dofs)
        #    temp[2+p] = patch_dofs
        #print rank, '\n', temp.T
