#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Initialise rascal.

"""

from importlib import metadata
try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError as e:
    raise ImportError(
        f"rascal is not setup or installed properly. {e}."
    ) from e

from . import calibrator, models, synthetic, util

__all__ = [
    "atlas",
    "calibrator",
    "houghtransform",
    "models",
    "plotting",
    "ransac",
    "sampler",
    "synthetic",
    "util",
]
