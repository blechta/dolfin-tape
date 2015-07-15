#!/usr/bin/env python

from distutils.core import setup
import re, os, glob

version = re.findall('__version__ = "(.*)"',
                     open('dolfintape/__init__.py', 'r').read())[0]

packages = [
    "dolfintape",
    "dolfintape.demo_problems",
    ]


CLASSIFIERS = """
Development Status :: 2 - Pre-Alpha
Environment :: Console
Intended Audience :: Science/Research
License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)
Programming Language :: Python
Topic :: Scientific/Engineering :: Mathematics
"""
classifiers = CLASSIFIERS.split('\n')[1:-1]

demofiles = glob.glob(os.path.join("demo", "*", "demo_*.py"))

setup(name="dolfintape",
      version=version,
      author="Jan Blechta",
      author_email="blechta@karlin.mff.cuni.cz",
      url="http://bitbucket.com/blechta/dolfin-tape",
      description="dolfin-tape, DOLFIN tools for a posteriori error estimation",
      long_description="tools for equilibrated-flux-reconstruction based "
                       "a posteriori error estimation methods",
      classifiers=classifiers,
      license="GNU LGPL v3 or later",
      packages=packages,
      package_dir={"dolfintape": "dolfintape"},
      package_data={"dolfintape": ["*.h"]},
      data_files=[(os.path.join("share", "dolfintape", os.path.dirname(f)), [f])
                  for f in demofiles],
    )
