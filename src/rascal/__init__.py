#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

from . import atlas
from . import calibrator
from . import houghtransform
from . import models
from . import plotting
from . import synthetic
from . import util

__all__ = [
    "atlas",
    "calibrator",
    "houghtransform",
    "models",
    "plotting",
    "synthetic",
    "util",
]
