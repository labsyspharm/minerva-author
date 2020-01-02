from setuptools import setup
from distutils.extension import Extension

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except:
    use_cython = False
else:
    use_cython = True

try:
    import numpy
except:
    raise Exception("Numpy is needed for installation")

import sys
import pytiff._version as _version

ext = ".pyx" if use_cython else ".cpp"

extensions = [
    Extension("pytiff._pytiff", ["pytiff/_pytiff.pyx"],
    libraries=["tiff"],
    include_dirs=["./pytiff", numpy.get_include()],
    language="c++",
    )]

if use_cython:
    extensions = cythonize(extensions, compiler_directives={'linetrace': True})

setup(
    ext_modules = extensions,
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "hypothesis"],
    name="pytiff",
    version=_version.__version__,
    packages=["pytiff", "pytiff._version"],
    package_dir={
        "pytiff" : "pytiff",
    },
    license="BSD",
    description="A libtiff wrapper to read tiled tiff images",
    long_description="Pytiff is a Python library for reading and writing large Tiff files. It supports tiled Tiffs and is able to read only a part of the image. While this is pytiffs main advantage, it also supports reading of many other image formats and writing of greyscale images in tile or scanline mode.",
    url="https://github.com/FZJ-INM1-BDA/pytiff",
    author="Big Data Analytics Group, INM-1, Research Center Juelich",
    author_email="p.glock@fz-juelich.de"
)
