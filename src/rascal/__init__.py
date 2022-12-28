#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Initialise rascal.

"""

import pkg_resources

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound as e:
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
