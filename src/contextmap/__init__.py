"""ContextMap2 public package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("contextmap")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
