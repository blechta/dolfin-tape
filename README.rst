===========================================================
dolfin-tape, DOLFIN tools for a posteriori error estimation
===========================================================

`dolfin-tape` provides tools for equilibrated-flux-reconstruction
based a posteriori error estimation methods built on top of
`DOLFIN <https://bitbucket.org/fenics-project/dolfin>`_, and
`the FEniCS project <http://fenicsproject.org>`_. Theoretical
references can be found within demo programs docstrings.

Dependencies
============

`dolfin-tape` needs

 * `DOLFIN <https://bitbucket.org/fenics-project/dolfin>`_
   compiled with with Python and PETSc.
 * PETSc, petsc4py
 * mpi4py

Installation
============

Run::

  python setup.py install [--user]

or just prepend to your PYTHONPATH by::

  source set_pythonpath.conf

Documentation
=============

The documentation currently consists only of this readme, class docstrings
and demo programs. This will likely be improved in the future.

Authors
=======

* Jan Blechta <blechta@karlin.mff.cuni.cz>

Copyright
=========

Copyright 2015-2016 Jan Blechta

License
=======

`dolfin-tape` is licensed under GNU Lesser General Public License version 3
or any later version. The full text of the license can be found in files
``COPYING`` and ``COPYING.LESSER``.

Contact
=======

`dolfin-tape` is hosted at https://github.com/blechta/dolfin-tape/
