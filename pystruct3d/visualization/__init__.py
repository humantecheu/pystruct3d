# SPDX-License-Identifier: MIT
# Copyright (c) 2023 HumanTech
import warnings

from pystruct3d.visualization.visualization import Visualizer


class Visualization(Visualizer):
    """Deprecated. Use :class:`Visualizer` instead."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        warnings.warn(
            "Visualization is deprecated; use Visualizer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = ["Visualization", "Visualizer"]
