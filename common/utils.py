import numpy as np

__all__ = ['la_index']

# TODO: What is the proper type for 64-bit PetscInt and how to determine it?
# NOTE: Seems that la_index is not mapped properly in DOLFIN SWIG wrappers,
#       see https://bitbucket.org/fenics-project/dolfin/issue/366
la_index = np.intc
