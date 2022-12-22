#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pkg_resources

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    raise ImportError(
        "rascal is not setup or installed properly. Unable to get version."
    )

from . import calibrator, models, synthetic, util

__all__ = [
    "calibrator",
    "models",
    "synthetic",
    "util",
]
