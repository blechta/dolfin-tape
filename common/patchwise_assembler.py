from dolfin import *
import numpy as np

from common import MatrixView, VectorView

parameters['graph_coloring_library'] = 'Boost'
#parameters['graph_coloring_library'] = 'Zoltan'

#mesh = UnitSquareMesh(6, 6)
mesh = UnitCubeMesh(6, 6, 6)

comm = mesh.mpi_comm()
rank = MPI.rank(comm)
size = MPI.size(comm)

vertex_colors = VertexFunction('size_t', mesh)
# TODO: These should give same result; which is cheaper?
vertex_colors.array()[:] = MeshColoring.color(mesh, np.array([0, 1, 0], dtype='uintp'))
#vertex_colors.array()[:] = MeshColoring.color(mesh, np.array([0, mesh.topology().dim(), 0], dtype='uintp'))
color_num = int(vertex_colors.array().max() - vertex_colors.array().min() + 1)

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


#c = Cell(mesh, 0)
#print 'cell', rank, c.global_index()
#v = Vertex(mesh, 0)
#print 'vertex', rank, v.global_index()
#f = Facet(mesh, 0)
#DistributedMeshTools.number_entities(mesh, 1)
#print 'facet', rank, f.global_index()

l = 2
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

patches_dim = tdim*dofs_per_facet*num_facets + (tdim + 1)*dofs_per_cell*num_cells
partitions_dim = [sum(patch_dim(v) for v in vertices(mesh) if vertex_colors[v] == p)
                  for p in range(color_num)]

assert sum(patch_dim(v) for v in vertices(mesh)) == patches_dim
#print rank, partitions_dim, patches_dim
assert sum(partitions_dim) == patches_dim

patches_dim_offset = MPI.global_offset(comm, patches_dim, True)
#print rank, patches_dim_offset, patches_dim

# Construct mapping of (rank-local) W dofs to (rank-global) patch-wise problems
num_dofs_with_ghosts = W.dofmap().tabulate_local_to_global_dofs().size # TODO: Is it obtainable cheaply?
dof_mapping = [np.empty(num_dofs_with_ghosts, dtype='uintp')
               for p in range(color_num)]
[dof_mapping[p].fill(-1) for p in range(color_num)]
dof_counter = patches_dim_offset
facet_dofs = [W.dofmap().tabulate_entity_dofs(tdim - 1, f)
              for f in range(tdim + 1)]
cell_dofs = W.dofmap().tabulate_entity_dofs(tdim, 0)
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
    #assert np.unique(local_dofs).tolist() == local_dofs # Does not hold now

    # Build global dofs
    num_dofs = patch_dim(v)
    assert num_dofs == len(local_dofs)
    global_dofs = np.arange(dof_counter, dof_counter + num_dofs, dtype='uintp')

    # Store mapping and increase counter
    dof_mapping[p][local_dofs] = global_dofs
    dof_counter += num_dofs
