"""
_version
Version information for pytiff.
"""

__all__ = [
    'VERSION_MAJOR',
    'VERSION_MINOR',
    'VERSION_PATCH',
    'VERSION_STRING',
    '__version__',
    '_package'
]

VERSION_MAJOR = 0
VERSION_MINOR = 8
VERSION_PATCH = 1

VERSION_STRING = "{major}.{minor}.{patch}".format(major=VERSION_MAJOR, minor=VERSION_MINOR, patch=VERSION_PATCH)

__version__ = VERSION_STRING
_package = __name__.split(".")[0]
