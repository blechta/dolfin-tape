from dolfin import *
import numpy as np
from petsc4py import PETSc

from common import MatrixView, VectorView, la_index

parameters['graph_coloring_library'] = 'Boost'
#parameters['graph_coloring_library'] = 'Zoltan'

mesh = UnitSquareMesh(2, 2)
#mesh = UnitCubeMesh(6, 6, 6)

comm = mesh.mpi_comm()
rank = MPI.rank(comm)
size = MPI.size(comm)

vertex_colors = VertexFunction('size_t', mesh)
# TODO: These should give same result; which is cheaper?
vertex_colors.array()[:] = MeshColoring.color(mesh, np.array([0, 1, 0], dtype='uintp'))
#vertex_colors.array()[:] = MeshColoring.color(mesh, np.array([0, mesh.topology().dim(), 0], dtype='uintp'))
# TODO: Isn't MPI.max needed here?!
color_num = int(vertex_colors.array().max() - vertex_colors.array().min() + 1)

max_uintp = int(np.uintp(-1))
cell_partitions = [CellFunction('size_t', mesh, max_uintp) for _ in xrange(color_num)]
for v in vertices(mesh):
    p = vertex_colors[v]
    for c in cells(v):
        cell_partitions[p][c] = p

plot(vertex_colors)
for cp in cell_partitions:
    plot(cp)
#interactive()


#c = Cell(mesh, 0)
#print 'cell', rank, c.global_index()
#v = Vertex(mesh, 0)
#print 'vertex', rank, v.global_index()
#f = Facet(mesh, 0)
#DistributedMeshTools.number_entities(mesh, 1)
#print 'facet', rank, f.global_index()

l = 0
RT = FunctionSpace(mesh, 'Raviart-Thomas', l+1)
DG = FunctionSpace(mesh, 'Discontinuous Lagrange', l)
W = RT*DG

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
    return num_interior_facets*dofs_per_facet + \
           num_patch_cells*dofs_per_cell # - 1 # factorization to zero mean

num_dofs_with_ghosts = W.dofmap().tabulate_local_to_global_dofs().size # TODO: Is it obtainable cheaply?
patches_dim = tdim*dofs_per_facet*num_facets + (tdim + 1)*dofs_per_cell*num_cells
partitions_dim = [sum(patch_dim(v) for v in vertices(mesh) if vertex_colors[v] == p)
                  for p in range(color_num)]

assert sum(patch_dim(v) for v in vertices(mesh)) == patches_dim
assert sum(partitions_dim) == patches_dim

patches_dim_offset = MPI.global_offset(comm, patches_dim, True)

# Construct mapping of (rank-local) W dofs to (rank-global) patch-wise problems
dof_mapping = [np.empty(num_dofs_with_ghosts, dtype=la_index)
               for p in range(color_num)]
[dof_mapping[p].fill(-1) for p in range(color_num)]
dof_counter = patches_dim_offset
facet_dofs = [W.dofmap().tabulate_entity_dofs(tdim - 1, f)
              for f in range(tdim + 1)]
cell_dofs = W.dofmap().tabulate_entity_dofs(tdim, 0) #[:-1] # factorization to zero mean
for v in vertices(mesh):
    p = vertex_colors[v]

    # Build local dofs
    local_dofs = []
    for c in cells(v):
        # TODO: Je to dobre? Je zde j spravny index facety?
        for j, f in enumerate(facets(c)):
            if f.incident(v): # Zero-flux on patch boundary
                local_dofs += W.dofmap().cell_dofs(c.index())[facet_dofs[j]].tolist()
        local_dofs += W.dofmap().cell_dofs(c.index())[cell_dofs].tolist()
    local_dofs = np.unique(local_dofs)

    # Build global dofs
    num_dofs = patch_dim(v)
    assert num_dofs == len(local_dofs)
    global_dofs = np.arange(dof_counter, dof_counter + num_dofs, dtype=la_index)

    # Store mapping and increase counter
    dof_mapping[p][local_dofs] = global_dofs
    dof_counter += num_dofs

assert dof_counter == patches_dim + patches_dim_offset

#print rank, dof_mapping
# TODO: validity of dof_mapping needs to be tested!

A = Matrix() # TODO: needs sparsity pattern!
b = Vector()

tl_A = TensorLayout(comm, np.array(2*(patches_dim,), dtype='uintp'), 0, 1,
       np.array(2*((patches_dim_offset, patches_dim_offset + patches_dim), ),
                dtype='uintp'), False)
tl_b = TensorLayout(comm, np.array((patches_dim,), dtype='uintp'), 0, 1,
       np.array(((patches_dim_offset, patches_dim_offset + patches_dim), ),
                dtype='uintp'), False)

# Empty map should do the same
#tl.local_to_global_map[0] = np.arange(patches_dim_offset,
#       patches_dim_offset + patches_dim, dtype='uintp')

# TODO: Sparsity patter missing!
#A.init(tl_A)
b.init(tl_b)
#as_backend_type(A).mat().setOption(
#    PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
# Enforce dropping of negative indices on VecSetValues
as_backend_type(b).vec().setOption(
    PETSc.Vec.Option.IGNORE_NEGATIVE_INDICES, True)

#A_patches = [MatrixView(A, W.dim(), W.dim(), dof_mapping[p], dof_mapping[p]) for p in range(color_num)]
#[assemble(inner(TrialFunction(W), TestFunction(W))*dx, tensor=A_patches[p], add_values=True) for p in range(color_num)]
b_patches = [VectorView(b, W.dim(), dof_mapping[p]) for p in range(color_num)]
#[assemble(TestFunctions(W)[1]*dx, tensor=b_patches[p], add_values=True) for p in range(color_num)]
[assemble(TestFunctions(W)[1]*dx, tensor=b_patches[p], add_values=True) for p in [0]]
as_backend_type(b).update_ghost_values()

print rank, 'mapping', [dof_mapping[p].size for p in range(color_num)]
print rank, 'b', b.size(), b.local_size(), b.local_range()
print rank, 'dofmap', W.dim(), W.dofmap().tabulate_local_to_global_dofs().size, W.dofmap().ownership_range()

w = Function(W)
#as_backend_type(w.vector()).vec().setOption(
#    PETSc.Vec.Option.IGNORE_NEGATIVE_INDICES, True)
#[b1.add_to_vector(w.vector()) for b1 in b_patches]
for b1 in b_patches:
    b1.add_to_vector(w.vector())
    b1.apply('add')
as_backend_type(w.vector()).update_ghost_values()
print rank, w.vector().array()
print rank, w.sub(1, deepcopy=True).vector().array()
plot(w.sub(1), interactive=True)
