from dolfin import not_working_in_parallel, facets, vertices

__all__ = ['mesh_diameter']

def mesh_diameter(mesh):
    # FIXME: Quadratic algorithm is too slow!
    """Return mesh diameter, i.e. \sup_{x,y \in mesh} |x-y|. Algorithm
    loops quadratically over boundary facets."""

    not_working_in_parallel("Function 'mesh_diameter'")
    assert mesh.topology().dim() == mesh.geometry().dim(), \
            "Function 'mesh_diameter' not working on manifolds."

    tdim = mesh.topology().dim()
    mesh.init(tdim-1, tdim)

    diameter = 0.0

    for f0 in facets(mesh):
        if not f0.exterior():
            continue

        for f1 in facets(mesh):
            if not f1.exterior():
                continue

            for v0 in vertices(f0):
                for v1 in vertices(f1):
                    diameter = max(diameter, v0.point().distance(v1.point()))

    return diameter
