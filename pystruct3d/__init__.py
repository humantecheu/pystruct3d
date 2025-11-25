from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pystruct3d")
except PackageNotFoundError:
    pass  # package is not installed or doesn't exist
