import numpy

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


extmod = [Extension("S_maximize_3q",["S_maximize_3q.pyx"],include_dirs=[numpy.get_include()])]#,qutip.get_include(),scipy.get_include()])]

setup(cmdclass = {'build_ext': build_ext},ext_modules = extmod)

#setup(cmdclass = {'build_ext': build_ext},ext_modules = [Extension("example", ["example.pyx"])])
