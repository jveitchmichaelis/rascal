#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

from . import calibrator
from . import models
from . import synthetic
from . import util

__all__ = [
    "calibrator",
    "models",
    "synthetic",
    "util",
]
