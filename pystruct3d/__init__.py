import logging
from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version

logging.getLogger(__name__).addHandler(logging.NullHandler())

with suppress(PackageNotFoundError):
    __version__ = version("pystruct3d")
