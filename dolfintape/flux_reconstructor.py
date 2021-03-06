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

from dolfin import *
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI as MPI4py

from itertools import chain

from dolfintape.tensor_views import MatrixView, VectorView, assemble
from dolfintape.utils import la_index_mpitype
from dolfintape.hat_function import hat_function_collection

__all__ = ['FluxReconstructor']

class FluxReconstructor(Variable):

    def __init__(self, mesh, degree):
        """Build flux reconstructor of given degree."""
        Variable.__init__(self)

        self._mesh = mesh

        # Color mesh vertices
        color_num, vertex_colors, cell_partitions = self._color_patches(mesh)
        self._color_num = color_num
        self._cell_partitions = cell_partitions

        # Build dof mapping to patches problem
        self._build_dof_mapping(degree, vertex_colors)

        # Prepare system tensors, tensor views and solver
        self._init_tensors()
        self._clear()
        self._assemble_matrix()
        self._setup_solver()

        # Prepare hat functions
        self._hat = [hat_function_collection(vertex_colors, p)
                     for p in xrange(color_num)]


    def dp(self, p):
        """Return volume measure on patches of color p."""
        return Measure("dx", domain=self._mesh, subdomain_id=1,
                       subdomain_data=self._cell_partitions[p])


    def L(self, p, D, f):
        # FIXME: Consolidate notation. It's very confusing, i.e different q here and below
        """Returns rhs linear form for flux reconstruction on patches p s.t.
        resulting flux q
          * reconstructs q ~ -D
          * equilibrates div(q) ~ f."""
        v, q = TestFunctions(self._W)
        hat = self._hat
        return ( -hat[p]*inner(D, v)
                 -hat[p]*f*q
                 +inner(grad(hat[p]), D)*q )*self.dp(p)

    def a(self, p):
        """Returns bilinear form for flux reconstruction on patches of color p.
        Mixed Poisson-like saddle-point form is used."""
        u, r = TrialFunctions(self._W)
        v, q = TestFunctions(self._W)
        return ( inner(u, v) - inner(r, div(v)) - inner(q, div(u)) )*self.dp(p)


    def reconstruct(self, *rhs_coefficients):
        # TODO: Add docstring!
        # Assemble rhs color by color
        t = Timer('dolfintape: assemble rhs for flux reconstruction')
        self._b.zero()
        for p in xrange(self._color_num):
            assemble(self.L(p, *rhs_coefficients), tensor=self._b_patches[p],
                     add_values=True, finalize_tensor=False)
        self._b.apply('add')
        t.stop()

        # Check if it is first solve
        if self.__dict__.get('_factored'):
            task = 'solve (second)'
        else:
            task = 'solve (first)'
            self._factored = True

        # Solve the system
        t = Timer('dolfintape: %s system for flux reconstruction' % task)
        assert self._solver, "Solver has not been initialized yet."
        try:
            self._solver.solve(self._x, self._b)
        except RuntimeError as e:
            t.stop()
            t = Timer('dolfintape: %s system for flux reconstruction'
                      ' with static pivotting' % task)

            # Enable static pivotting plus two iterative refinement steps
            opts = self.options()
            opts.setValue('mat_mumps_cntl_4', 1e-6)
            opts.setValue('mat_mumps_icntl_10', -2)
            self._solver.ksp().setFromOptions()

            # Try again
            self._solver.solve(self._x, self._b)
        t.stop()

        t = Timer('dolfintape: collect patch-wise flux reconstruction')
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
        t.stop()

        return w


    @staticmethod
    def _color_patches(mesh):
        """Returns number of colors, vertex function with colors and sequence
        of cell functions cp s.t.:
            cp[p][c] = 1 ... if c belongs to patch of color p
            cp[p][c] = 0 ... otherwise
        """
        t = Timer('dolfintape: color patches')
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

        t.stop()

        return color_num, vertex_colors, cell_partitions


    def _assemble_matrix(self):
        """Assembles flux reconstruction matrix on all the patches.
        The matrix is stored as self._A but accessed by assembler through
        matrix views self._A_patches[p] on patches of color p."""
        t = Timer('dolfintape: assemble matrix for flux reconstruction')

        # Assemble matrix color by color
        # NOTE: assuming self._A is zeroed
        for p in xrange(self._color_num):
            assemble(self.a(p), tensor=self._A_patches[p],
                     add_values=True, finalize_tensor=False)
        self._A.apply('add')

        # Ensure that this method is not called twice (to avoid zeroing matrix)
        def on_second_call():
            raise RuntimeError("_assemble_matrix can't be called twice!")
        self._assemble_matrix = lambda *args, **kwargs: on_second_call()

        t.stop()


    def _setup_solver(self):
        """Initilize Cholesky solver for solving flux reconstruction."""
        t = Timer('dolfintape: setup solver for flux reconstruction')
        opts = self.options()

        # Diagnostic output
        #opts.setValue('mat_mumps_icntl_4', 2)
        #opts.setValue('mat_mumps_icntl_2', 6)

        # Ordering options
        #opts.setValue('mat_mumps_icntl_7', 7)
        #opts.setValue('mat_mumps_icntl_12', 0)

        # Pivotting threshold
        #opts.setValue('mat_mumps_cntl_1', 1e-0)

        # Static pivotting plus one iterative refinement step
        #opts.setValue('mat_mumps_cntl_4', 1e-6)
        #opts.setValue('mat_mumps_icntl_10', -1)

        self._solver = solver = PETScLUSolver('mumps')
        #solver.set_options_prefix(opts.prefix) # Buggy
        solver.set_operator(self._A)

        # Allow Cholesky only with PETSc having fixed
        # https://bitbucket.org/petsc/petsc/issue/81
        if PETSc.Sys.getVersion() >= (3, 5, 3):
            solver.parameters['symmetric'] = True

        # NOTE: KSPSetFromOptions is already called in constructor!
        solver.ksp().setOptionsPrefix(opts.prefix)
        solver.ksp().setFromOptions()

        t.stop()


    def options(self):
        """Return PETSc Options database for all owned PETSc objects
        """
        opts = self.__dict__.get('_opts')
        if not opts:
            prefix = 'dolfin_%s_' % self.id()
            opts = self._opts = PETSc.Options(prefix)
        return opts


    def __del__(self):
        # Remove unused PETSc options to avoid database overflow
        # NOTE: Rewrite maybe needed with PETSc 3.7, see
        # https://bitbucket.org/petsc/petsc4py/commits/c6668262ee0af6e186e9f51641ae53bba3691be2
        opts = self.options()
        for k in opts.getAll().keys():
            opts.delValue(k)


    def get_info(self):
        """Return info about space dimensions, system matrix
        and factor matrix"""
        info_mat = self._A.mat().getInfo()
        info_fact = self._solver.ksp().getPC().getFactorMatrix().getInfo()
        info = {"dim_full": self._W.dim(),
                "dim_agg": self._A.size(0),
                "mat_info": info_mat,
                "fact_info": info_fact,}
        return info


    def pc_view(self):
        """Call PETSc PCView on underlying PC object"""
        solver = self._solver
        ksp = solver.ksp()
        pc = ksp.getPC()
        pc.view()


    def _init_tensors(self):
        """Prepares matrix, rhs and solution vector for the system on patches
        plus views into them living on usual function space allowing for
        assembling into system tensors on partitions color by color and
        querying for solution. Local to global map and ghosts are not
        initialized for the system tensors as only indexing by global indices
        is used from Vector/MatrixView."""
        t = Timer('dolfintape: init tensors for flux reconstruction')

        comm = self._mesh.mpi_comm()
        color_num = self._color_num
        patches_dim_owned = self._patches_dim_owned
        dof_mapping = self._dof_mapping
        local_to_global_patches = self._local_to_global_patches

        # Prepare tensor layout and build sparsity pattern
        im = IndexMap(comm, patches_dim_owned, 1)
        im.set_local_to_global(local_to_global_patches)
        tl_A = TensorLayout(comm, [im, im], 0,
                            TensorLayout.Sparsity_SPARSE,
                            TensorLayout.Ghosts_UNGHOSTED)
        tl_b = TensorLayout(comm, [im], 0,
                            TensorLayout.Sparsity_DENSE,
                            TensorLayout.Ghosts_UNGHOSTED)
        pattern = tl_A.sparsity_pattern()
        pattern.init(comm, [im, im])
        self._build_sparsity_pattern(pattern)

        # Init tensors
        self._A = A = PETScMatrix()
        self._b = b = PETScVector()
        self._x = x = PETScVector()
        A.init(tl_A)
        b.init(tl_b)
        x.init(tl_b)

        # Set unique options prefix for objects
        prefix = self.options().prefix
        A.mat().setOptionsPrefix(prefix)
        b.vec().setOptionsPrefix(prefix)
        x.vec().setOptionsPrefix(prefix)

        # Drop assembled zeros as our sparsity pattern counts for non-zeros only
        A.mat().setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)

        # Enforce dropping of negative indices on Vec(Get|Set)Values
        b.vec().setOption(PETSc.Vec.Option.IGNORE_NEGATIVE_INDICES, True)
        x.vec().setOption(PETSc.Vec.Option.IGNORE_NEGATIVE_INDICES, True)

        # Initialize tensor views to partitions with color p
        self._A_patches = [MatrixView(A, self._W.dim(), self._W.dim(),
                                      dof_mapping[p], dof_mapping[p])
                           for p in xrange(color_num)]
        self._b_patches = [VectorView(b, self._W.dim(), dof_mapping[p])
                           for p in xrange(color_num)]
        self._x_patches = [VectorView(x, self._W.dim(), dof_mapping[p])
                           for p in xrange(color_num)]

        t.stop()


    def _build_sparsity_pattern(self, pattern):
        t = Timer('dolfintape: build sparsity for flux reconstruction')

        mesh = self._mesh
        comm = self._mesh.mpi_comm()
        cell_dofs_0 = self._W.sub(0).dofmap().cell_dofs
        cell_dofs_1 = self._W.sub(1).dofmap().cell_dofs
        color_num = self._color_num
        dof_mapping = self._dof_mapping
        patches_dim_offset = self._patches_dim_offset
        patches_dim_owned = self._patches_dim_owned
        local_to_global_patches = self._local_to_global_patches

        # Build inverse of local_to_global_patches
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
        global_to_local_patches = np.vectorize(global_to_local_patches,
                                               otypes=[la_index_dtype()])

        # Build sparsity pattern for mixed Laplacian
        for c in cells(mesh):
            RT_dofs = cell_dofs_0(c.index())
            DG_dofs = cell_dofs_1(c.index())
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
                    DG_dofs_patch = global_to_local_patches(DG_dofs_patch)
                    RT_dofs_patch = global_to_local_patches(RT_dofs_patch)
                    RT_dofs_patch_unowned = global_to_local_patches(RT_dofs_patch_unowned)
                    pattern.insert_local([RT_dofs_patch_unowned, RT_dofs_patch])
                    pattern.insert_local([RT_dofs_patch_unowned, DG_dofs_patch])
        pattern.apply()

        t.stop()


    def _build_dof_mapping(self, degree, vertex_colors):
        t = Timer('dolfintape: build dofmap for flux reconstruction')

        mesh = self._mesh
        color_num = self._color_num

        comm = mesh.mpi_comm()
        rank = MPI.rank(comm)
        size = MPI.size(comm)

        # Construct cell space for flux reconstruction mixed problem
        RT = FiniteElement('Raviart-Thomas', mesh.ufl_cell(), degree + 1)
        DG = FiniteElement('Discontinuous Lagrange', mesh.ufl_cell(), degree)
        W = FunctionSpace(mesh, RT*DG)
        self._W = W
        dofmap = W.dofmap()
        dofmap_dg = W.sub(1).dofmap()
        tdim = mesh.topology().dim()

        # Function for checking if a vertex is on domain boundary
        mesh.init(tdim-1, tdim)
        def on_boundary(vertex):
            return any(f.exterior() for f in facets(vertex))

        # Function for checking whether one DG dof should be removed at patch
        def remove_dg_dof(vertex):
            # True if vertex is interior but only on one rank if patch is shared
            # TODO: Is there some other scheme than "shared on lowest rank"?
            return not on_boundary(vertex) and \
                   ( not vertex.is_shared()
                     or rank <= min(vertex.sharing_processes()) )

        # Pick some dimensions
        num_cells = mesh.num_cells()
        num_facets = mesh.num_facets()
        dofs_per_vertex = dofmap.num_entity_dofs(0)
        dofs_per_facet = dofmap.num_entity_dofs(tdim-1)
        dofs_per_cell = dofmap.num_entity_dofs(tdim)
        assert dofs_per_vertex == 0
        num_dofs_with_ghosts = dofmap.index_map().size(IndexMap.MapSize_ALL)

        # Local dimension of patch space
        patches_dim = tdim*dofs_per_facet*num_facets + (tdim + 1)*dofs_per_cell*num_cells
        # Remove one DG DOF per interior patch
        patches_dim -= sum(1 for v in vertices(mesh) if remove_dg_dof(v))
        self._patches_dim = patches_dim

        # Construct mapping of (rank-local) W dofs to (rank-global) patch-wise problems
        self._dof_mapping = dof_mapping = [np.empty(num_dofs_with_ghosts, dtype=la_index_dtype())
                       for p in xrange(color_num)]
        [dof_mapping[p].fill(-1) for p in xrange(color_num)]
        dof_counter = 0
        facet_dofs = [dofmap.tabulate_entity_dofs(tdim - 1, f)
                      for f in range(tdim + 1)]
        cell_dofs = dofmap.tabulate_entity_dofs(tdim, 0)
        local_ownership_size = dofmap.index_map().size(IndexMap.MapSize_OWNED)
        local_to_global = dofmap.local_to_global_index
        shared_nodes = dofmap.shared_nodes()
        shared_dofs = {r: {} for r in sorted(dofmap.neighbours())}
        for v in vertices(mesh):
            p = vertex_colors[v]

            # Build local dofs
            local_dofs = []
            for c in cells(v):
                c_dofs = dofmap.cell_dofs(c.index())
                for j, f in enumerate(facets(c)):
                    if f.incident(v): # Zero-flux on patch boundary
                        local_dofs += c_dofs[facet_dofs[j]].tolist()
                local_dofs += c_dofs[cell_dofs].tolist()

            # Remove one DG DOF per interior patch
            if remove_dg_dof(v):
                removed_dof = local_dofs.pop()
                assert removed_dof in dofmap_dg.cell_dofs(c.index())

            # Exclude unowned DOFs
            local_dofs = np.unique([dof for dof in local_dofs if dof < local_ownership_size])
            assert local_dofs.min() >= 0 and local_dofs.max() < local_ownership_size

            # Build global dofs
            num_dofs = local_dofs.size
            global_dofs = np.arange(dof_counter, dof_counter + num_dofs, dtype=la_index_dtype())

            # Store mapping and increase counter
            dof_mapping[p][local_dofs] = global_dofs
            dof_counter += num_dofs

            # Prepare shared dofs
            global_vertex_index = None
            for dof in local_dofs:
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

        assert dof_counter==patches_dim if size==1 else dof_counter<=patches_dim

        # Compute patches ownership
        self._patches_dim_owned = patches_dim_owned = dof_counter
        self._patches_dim_offset = patches_dim_offset \
                = MPI.global_offset(comm, patches_dim_owned, True)

        # Switch to global indexing
        for p in xrange(color_num):
            dof_mapping[p][dof_mapping[p]>=0] += patches_dim_offset
        for d1 in shared_dofs.itervalues():
            for l in d1.itervalues():
                assert len(l) == 2*tdim
                for i in range(len(l)/2):
                    l[2*i+1] += patches_dim_offset

        # Prepare data for MPI communication
        c = chain.from_iterable(chain([v], data)
                                for r in sorted(shared_dofs.iterkeys())
                                for v, data in shared_dofs[r].iteritems())
        sendbuf = np.fromiter(c, dtype=la_index_dtype())
        num_shared_unowned = dofmap.index_map().size(IndexMap.MapSize_UNOWNED)
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

        # Maps from unowned local patch dof to global patch dof; Cp': ip' -> Ip
        self._local_to_global_patches = local_to_global_patches \
                = np.empty(patches_dim-patches_dim_owned, dtype='uintp')

        # Add unowned DOFs to dof_mapping
        dof_counter = 0
        mesh.init(tdim-1, tdim)
        for c in cells(mesh):

            # Loop over facets on rank interface
            for i, f in enumerate(facets(c)):
                if f.num_entities(tdim) == 2 or f.exterior():
                    continue

                # Get unonwed facet dofs
                c_dofs =  dofmap.cell_dofs(c.index())
                f_dofs = [dof for dof in c_dofs[facet_dofs[i]]
                              if dof>=local_ownership_size]
                if len(f_dofs) == 0:
                    continue

                for dof in f_dofs:
                    # Get global dof index
                    global_dof = local_to_global(dof)

                    # Get owner
                    ranks = shared_nodes[dof]
                    assert ranks.size == 1
                    owner = ranks[0]

                    # Search for received data with matching global index
                    # TODO: Avoid copying large portion of recvbuf on and on
                    owner_recvbuf_range = slice(
                            recvdispls[owner],
                            recvdispls[owner] + recvcounts[owner],
                            2*tdim+1 )
                    global_dofs = recvbuf[owner_recvbuf_range]
                    j = np.where(global_dofs == global_dof) # TODO: THIS MAY YIELD QUADRATIC SCALING !
                    assert len(j)==1 and j[0].size==1
                    j = owner_recvbuf_range.step * j[0][0] + owner_recvbuf_range.start
                    assert recvbuf[j] == global_dof

                    # Get vertex (patch) indices and patch dofs
                    vertex_indices = recvbuf[j+1:j+1+2*tdim:2]
                    patch_dofs     = recvbuf[j+2:j+2+2*tdim:2]

                    # Loop over vertices and store the data
                    for v in vertices(f):
                        p = vertex_colors[v]
                        k = np.where(vertex_indices==v.global_index())
                        assert len(k)==1 and k[0].size==1
                        k = k[0][0]
                        assert 0 <= k < tdim

                        # Store received dato to patch dof mapping
                        dof_mapping[p][dof] = patch_dofs[k]

                        # Also construct (arbitrarily - as we have not defined
                        # local patch indices so far) local to global
                        local_to_global_patches[dof_counter] = patch_dofs[k]

                        dof_counter += 1

        assert dof_counter == patches_dim - patches_dim_owned

        # TODO: Consider sorting local_to_global_patches
        # TODO: validity of dof_mapping needs to be tested!

        t.stop()


    def _clear(self):
        """Clears objects needed only for initialization of self."""
        # TODO: Remove this and hadle variables lifetime by proper scoping
        del self._local_to_global_patches

        # This is not really deleted now as it is referenced by tensor views
        del self._dof_mapping

        # TODO: Clear unneeded mesh connectivity
