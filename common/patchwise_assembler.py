from dolfin import *
import numpy as np

parameters['graph_coloring_library'] = 'Boost'
#parameters['graph_coloring_library'] = 'Zoltan'

mesh = UnitSquareMesh(6, 6)

vertex_colors = VertexFunction('size_t', mesh)
# TODO: These should give same result; which is cheaper?
vertex_colors.array()[:] = MeshColoring.color(mesh, np.array([0, 1, 0], dtype='uintp'))
#vertex_colors.array()[:] = MeshColoring.color(mesh, np.array([0, mesh.topology().dim(), 0], dtype='uintp'))
color_num = int(vertex_colors.array().max() - vertex_colors.array().min() + 1)

int_limit = 2**63 - 1
cell_partitions = [CellFunction('size_t', mesh, int_limit) for _ in xrange(color_num)]
for v in vertices(mesh):
    p = vertex_colors[v]
    for c in cells(v):
        cell_partitions[p][c] = p

plot(vertex_colors)
for cp in cell_partitions:
    plot(cp)
interactive()
