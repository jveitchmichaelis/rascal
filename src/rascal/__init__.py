#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass  # package is not installed

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
