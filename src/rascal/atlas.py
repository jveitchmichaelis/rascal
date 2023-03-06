#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Atlas class for handling arc lines.

"""

import logging
import os
import time
from collections import Counter
from dataclasses import MISSING, dataclass, field, fields
from pprint import pprint
from typing import List, Optional, Union

import numpy as np
import yaml
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)
from .util import load_calibration_lines, vacuum_to_air_wavelength


@dataclass(kw_only=True)
class AtlasLine:
    """
    For storing information of an atlas line

    """

    wavelength: float
    element: str
    intensity: float
    source: str


@dataclass(kw_only=True)
class AtlasLine:
    """
    For storing basic information of an atlas line

    """

    wavelength: float
    element: str
    intensity: float = 0
    source: str

    def __repr__(self):
        if self.intensity > 0:
            return f"{self.element} ({self.source}): {self.wavelength} Å, {self.intensity}"
        else:
            return f"{self.element} ({self.source}): {self.wavelength} Å"


@dataclass(kw_only=True)
class Atlas:
    """
    Creates an atlas of arc lines.

    Arc lines are taken from a general list of NIST lines and can be
    filtered using the minimum relative intensity (note this may not be
    accurate due to instrumental effects such as detector response,
    dichroics, etc) and minimum line separation.

    Lines are filtered first by relative intensity, then by separation.
    This is to improve robustness in the case where there is a strong
    line very close to a weak line (which is within the separation limit).

    The vacuum to air wavelength conversion is deafult to False because
    observatories usually provide the line lists in the respective air
    wavelength, as the corrections from temperature and humidity are
    small. See https://emtoolbox.nist.gov/Wavelength/Documentation.asp

    """

    line_list: str = "nist"
    min_wavelength: float = MISSING
    max_wavelength: float = MISSING
    range_tolerance: float = 500.0
    min_intensity: float = 10.0
    min_distance: float = 10.0
    brightest_n_lines: int = 100
    vacuum: bool = False
    pressure: float = 101325.0
    temperature: float = 273.15
    relative_humidity: float = 0.0
    elements: Optional[List[str]] = MISSING

    wavelengths: Optional[List[float]] = field(default=None)
    intensities: Optional[List[float]] = field(default=None)

    def __post_init__(self):

        if isinstance(self.elements, str):
            self.elements = [self.elements]

        self.atlas_lines = []

        if self.line_list == "manual":
            self.add_manual()
        elif self.line_list == "nist":
            self.add_list(self.line_list)
        else:
            raise NotImplementedError

    def add_manual(self):

        # Case when we have a single wavelength
        if not isinstance(self.wavelengths, list):
            self.wavelengths = list(self.wavelengths)

        # If a single element is provided, assume that
        # all lines are from this element
        if len(self.elements) == 1:
            self.elements = self.elements * len(self.wavelengths)

        # Empty intensity
        if self.intensities is None:
            self.intensities = [0] * len(self.wavelengths)
        elif not isinstance(self.intensities, list):
            self.intensities = list(self.intensities)

        assert len(self.elements) == len(self.wavelengths), ValueError(
            "Input elements and wavelengths have different length."
        )
        assert len(self.elements) == len(self.intensities), ValueError(
            "Input elements and intensities have different length."
        )

        if self.vacuum:

            self.wavelengths = vacuum_to_air_wavelength(
                self.wavelengths,
                self.temperature,
                self.pressure,
                self.relative_humidity,
            )

        for element, wavelength, intensity in list(
            zip(self.elements, self.wavelengths, self.intensities)
        ):
            if wavelength < (self.min_wavelength - self.range_tolerance):
                logger.warning(
                    f"User-supplied wavelength {wavelength} is below the minimum atlas wavelength of {self.min_wavelength} with a tolerance of {self.range_tolerance}, will not use."
                )
                continue

            if wavelength > (self.max_wavelength + self.range_tolerance):
                logger.warning(
                    f"User-supplied wavelength {wavelength} is above the maximum atlas wavelength of {self.max_wavelength} with a tolerance of {self.range_tolerance}, will not use."
                )
                continue

            self.atlas_lines.append(
                AtlasLine(
                    wavelength=wavelength,
                    element=element,
                    intensity=intensity,
                    source="user",
                )
            )

    def add_list(self, line_list):
        min_atlas_wavelength = self.min_wavelength - self.range_tolerance

        max_atlas_wavelength = self.max_wavelength + self.range_tolerance

        for element in self.elements:

            (
                atlas_elements_tmp,
                atlas_tmp,
                atlas_intensities_tmp,
            ) = load_calibration_lines(
                elements=element,
                linelist=line_list,
                min_atlas_wavelength=min_atlas_wavelength,
                max_atlas_wavelength=max_atlas_wavelength,
                min_intensity=self.min_intensity,
                min_distance=self.min_distance,
                brightest_n_lines=self.brightest_n_lines,
                vacuum=self.vacuum,
                pressure=self.pressure,
                temperature=self.temperature,
                relative_humidity=self.relative_humidity,
            )

            for element, wavelength, intensity in list(
                zip(atlas_elements_tmp, atlas_tmp, atlas_intensities_tmp)
            ):
                self.atlas_lines.append(
                    AtlasLine(
                        wavelength=wavelength,
                        element=element,
                        intensity=intensity,
                        source=self.line_list,
                    )
                )

    @classmethod
    def from_config(cls, config):
        if isinstance(config, str):
            return cls(OmegaConf.load(config))
        elif isinstance(config, dict):
            return cls(OmegaConf.create(config))
        else:
            raise NotImplementedError

    def get_elements(self):
        """
        Returns a list of per-line elements in the atlas

        Returns
        -------
        element_list: list

        """

        element_list = [line.element for line in self.atlas_lines]

        return element_list

    def get_lines(self):
        """
        Returns a list of line wavelengths in the atlas

        Returns
        -------
        wavelength_list: list

        """
        return [line.wavelength for line in self.atlas_lines]

    def get_intensities(self):
        """
        Returns a list of per-line intensities in the atlas

        Returns
        -------
        intensity_list: list

        """

        intensity_list = [line.intensity for line in self.atlas_lines]

        return intensity_list

    def get_sources(self):
        """
        Returns a list of per-line source in the atlas

        Returns
        -------
        source_list: list

        """

        source_list = [line.source for line in self.atlas_lines]

        return source_list

    def __len__(self):
        return len(self.atlas_lines)


@dataclass
class AtlasCollection:
    atlases: List[Atlas] = MISSING
    exclude_wavelengths: List[float] = field(default_factory=list)
    exclude_elements: List[str] = field(default_factory=list)
    exclude_tolerance: float = 10.0
    min_wavelength: Optional[float] = 0
    max_wavelength: Optional[float] = 1e9

    @classmethod
    def from_config(self, config):
        return self(OmegaConf.load(config).atlases)

    def __post_init__(self):
        self.atlases = [Atlas(**config) for config in self.atlases]

    def line_valid(self, line):
        if line.wavelength < self.min_wavelength:
            return False
        elif line.wavelength > self.max_wavelength:
            return False
        elif line.element in self.exclude_elements:
            return False

        for wavelength in self.exclude_wavelengths:
            if abs(line.wavelength - wavelength) < self.exclude_tolerance:
                return False

        return True

    @property
    def atlas_lines(self):
        lines = []

        for atlas in self.atlases:
            lines.extend(atlas.atlas_lines)

        lines = list(filter(self.line_valid, lines))

        return sorted(lines, key=lambda x: x.wavelength)
