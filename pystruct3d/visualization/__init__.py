import warnings

from pystruct3d.visualization.visualization import Visualizer


class Visualization(Visualizer):
    """Deprecated. Use :class:`Visualizer` instead."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Visualization is deprecated; use Visualizer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = ["Visualization", "Visualizer"]
