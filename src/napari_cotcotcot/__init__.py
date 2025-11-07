"""napari-cotcotcot: A napari plugin for CoTracker-based tracking."""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._widget import cotracker_widget

__all__ = ("cotracker_widget",)
