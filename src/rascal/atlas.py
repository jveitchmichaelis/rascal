#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Atlas class for handling arc lines.

"""

import csv
import logging
import os
import time
from collections import Counter
from dataclasses import MISSING, dataclass, field, fields
from enum import Enum, auto
from glob import glob
from importlib import resources as import_resources
from pprint import pprint
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from omegaconf import OmegaConf

from .util import vacuum_to_air_wavelength

logger = logging.getLogger(__name__)


class LineSource(Enum):
    NIST_STRONG = auto()
    NIST_ALL = auto()


# @dataclass(kw_only=True)
@dataclass
class AtlasLine:
    """
    For storing basic information of an atlas line

    """

    wavelength: float
    element: str
    source: str = ""
    intensity: float = 0

    def __repr__(self):
        if self.intensity > 0:
            return f"{self.element} ({self.source}): {self.wavelength} Å, {self.intensity}"
        else:
            return f"{self.element} ({self.source}): {self.wavelength} Å"


# @dataclass(kw_only=True)
@dataclass
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

    min_wavelength: float = MISSING
    max_wavelength: float = MISSING
    elements: Optional[Any] = None
    line_list: str = "nist"
    range_tolerance: float = 0.0
    min_intensity: Optional[float] = 10.0
    min_distance: float = 10.0
    brightest_n_lines: Optional[int] = field(default=None)
    vacuum: bool = False
    pressure: float = 101325.0
    temperature: float = 273.15
    relative_humidity: float = 0.0
    wavelengths: Optional[List[float]] = field(default=None)
    intensities: Optional[List[float]] = field(default=None)
    use_accurate_lines: Optional[bool] = True
    atlas_lines: Optional[List[AtlasLine]] = field(default=None)
    nist_source: LineSource = LineSource.NIST_STRONG

    def __post_init__(self):

        if isinstance(self.elements, str):
            self.elements = [self.elements]

        self.min_atlas_wavelength = self.min_wavelength - self.range_tolerance
        self.max_atlas_wavelength = self.max_wavelength + self.range_tolerance
        self.atlas_lines = []

        logger.info(
            f"Loading lines from {self.line_list} list between {self.min_atlas_wavelength} and {self.max_atlas_wavelength} for elements: {set(self.elements)}"
        )
        logger.info(
            f"Filtering lines by intensity > {self.min_intensity} and separation > {self.min_distance} Å"
        )

        if self.line_list == "manual":
            self.add_manual()
        elif self.line_list == "nist":
            self.add_nist()
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
                    f"User-supplied wavelength {wavelength} is below the "
                    f"minimum atlas wavelength of {self.min_wavelength} with "
                    f"a tolerance of {self.range_tolerance}, will not use."
                )
                continue

            if wavelength > (self.max_wavelength + self.range_tolerance):
                logger.warning(
                    f"User-supplied wavelength {wavelength} is above the "
                    f"maximum atlas wavelength of {self.max_wavelength} with "
                    f"a tolerance of {self.range_tolerance}, will not use."
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

    def add_nist(self):
        assert len(self.elements) == 1

        s = self.elements[0].split(" ")

        if len(s) == 2:
            states = s[1]
        else:
            states = []

        self.add_lines(
            nist_lines(
                element=s[0],
                states=states,
                source=self.nist_source,
                only_accurate=self.use_accurate_lines,
            ),
            source="nist",
            vacuum=True,
        )

    """
    def filter_lines(self, lines : List[AtlasLine], min_wavelength=3500, max_wavelength=8200, top_k=20):
        
        wavelengths = [line.wavelength for line in lines]
        wave_mask = filter_waves(wavelengths)
        
        # Filter intensities
        intensities = np.array([l['intensity'] for l in data_dict if l['ionisation'] == state])
        intensity_sort = np.argsort(intensities[wave_mask])[::-1]

        # Take top_k intensities 
        wavelengths = wavelengths[wave_mask][intensity_sort][:top_k]
        intensities = intensities[wave_mask][intensity_sort][:top_k]

        # Re-sort by wavelength
        wavelength_sort = np.argsort(wavelengths)
        wavelengths = wavelengths[wavelength_sort]
        intensities = intensities[wavelength_sort]

        # Filter by distance
        distance_idx = filter_distance(wavelengths, 40)

        wavelengths = wavelengths[distance_idx]
        intensities = intensities[distance_idx]
        
        return wavelengths
    """

    @classmethod
    def from_config(cls, config):
        if isinstance(config, str):
            return cls(OmegaConf.load(config))
        elif isinstance(config, dict):
            return cls(OmegaConf.create(config))
        else:
            raise NotImplementedError

    def add_lines(
        self,
        lines: List[AtlasLine],
        source: str = "manual",
        vacuum: bool = True,
    ):
        """Add lines to the atlas

        Parameters
        ----------
        lines : _type_
            _description_
        source : str, optional
            Source string, used for visualisation/reference, by default "manual"
        vacuum : bool, optional
            Whether the provided lines are in vacuum wavelengths or not, by default True
        """

        for line in lines:

            if line["wavelength"] < self.min_atlas_wavelength:
                continue
            elif line["wavelength"] > self.max_atlas_wavelength:
                continue

            if "intensity" in line and line["intensity"] < self.min_intensity:
                continue
            else:
                line["intensity"] = 0

            if not self.vacuum and vacuum:
                line["wavelength"] = float(
                    vacuum_to_air_wavelength(
                        line["wavelength"],
                        temperature=self.temperature,
                        pressure=self.pressure,
                        relative_humidity=self.relative_humidity,
                    )
                )

            self.atlas_lines.append(
                AtlasLine(
                    wavelength=line["wavelength"],
                    element=line["element"],
                    intensity=line["intensity"],
                    source=source,
                )
            )

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
        """Create an atlas collection from a yaml configuration file or dictionary. The config
        object (or file) must have an "atlases" key that contains a list of atlas configurations.

        For example:

        ```
        atlases:
          -
            elements: ['Hg']
            min_wavelength: 4000
            max_wavelength: 7000
            min_intensity: 50
          -
            elements: ['Ar']
            min_wavelength: 7000
            max_wavelength: 9000
            min_intensity: 50
        ```

        Parameters
        ----------
        config : Union[str, dict]
            Configurations for atlases.

        Returns
        -------
        AtlasCollection

        Raises
        ------
        NotImplementedError
            If the provided config is not supported
        """

        if isinstance(config, str):
            return self(OmegaConf.load(config).atlases)
        elif isinstance(config, dict):
            return self(OmegaConf.create(config).atlases)
        else:
            raise NotImplementedError

    def __post_init__(self):
        """Runs after initialisation and loads Atlases from the provided configurations"""

        for idx, config in enumerate(self.atlases):
            self.atlases[idx] = Atlas(**config)

    def line_valid(self, line):
        """Check if a line meets the filter requirements (within wavelength, non-excluded element)

        Parameters
        ----------
        line : AtlasLine
            Atlas line to check

        Returns
        -------
        bool:
            Line validity
        """
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
        """Atlas lines in this atlas sorted by wavelength

        Returns
        -------
        List[AtlasLine]
            List of atlas lines in the atlas, to pass to a calibrator
        """
        lines = []

        for atlas in self.atlases:
            lines.extend(atlas.atlas_lines)

        lines = list(filter(self.line_valid, lines))

        return sorted(lines, key=lambda x: x.wavelength)

    @property
    def wavelengths(self):
        """Wavelengths in this atlas

        Returns
        -------
        List[float]
            List of wavelengths
        """
        return [line.wavelength for line in self.atlas_lines]


def nist_files(
    element: str,
    states: List[str] = ["I", "II"],
    source: LineSource = LineSource.NIST_STRONG,
) -> List[str]:
    """
    Locate atlas files for a particular element and an optional state. By default, only the I and II
    ionisation states are returned (as these are among the more commonly used emission lines). Rascal
    provides line lists up to the V state. Lines are sourced from either the full NIST reference
    catalogue, or from the "strong" line lists also published by NIST (default).

    Parameters
    ----------
    element: str
        Element to look up
    states: List[str]
        List of ionisation states, defaults to ['I', 'II']
    source: LineSource
        Preferred line source (NIST_ALL or NIST_STRONG)

    Returns
    -------
    List(str):
        List of files containing reference lines

    """

    line_files = []

    get_ref = lambda path: (import_resources.files(__package__) / path)

    if source == LineSource.NIST_ALL:
        root = f"arc_lines/nist_clean_"
    elif source == LineSource.NIST_STRONG:
        root = f"arc_lines/strong_lines/"
    else:
        raise NotImplementedError(
            f"This function should only be used to load lines from NIST reference lists."
        )

    root = os.path.join(import_resources.files(__package__), root)

    if len(states) == 0:
        line_files = glob(root + f"{element}_*.csv")
    else:
        for state in states:
            file_name = root + f"{element}_{state}.csv"
            if os.path.exists(file_name):
                line_files.append(file_name)

    return line_files


def open_line_list(
    path: str, element: Optional[str] = None, accurate=False
) -> List[Dict]:
    """
    Load a line list from a CSV file. The file must contain the following fields:

    - element
    - intensity
    - wavelength

    If element is provided, then only entries with a matching prefix will be returned,
    otherwise all lines in the file will be returned.

    Parameters
    ----------
    path: str
        path to line file
    element: optional, str
        element to filter

    Returns
    -------
    lines: List[Dict]
        list of line dictionaries which minimally contain the fields above

    """

    with open(path) as fp:
        lines_raw = fp.readlines()
        line_reader = csv.DictReader(lines_raw)

    # Validate line list columns:
    assert "element" in line_reader.fieldnames
    assert "intensity" in line_reader.fieldnames
    assert "wavelength" in line_reader.fieldnames

    line_reader = list(line_reader)

    lines = []
    for line in line_reader:

        if element is not None:
            if not line["element"].startswith(element):
                continue

        if accurate and line["acc"] == "":
            continue

        lines.append(line)

    if len(lines) == 0:
        logger.warning("Empty line list.")

    # Co-erce types:
    for line in lines:
        line["wavelength"] = float(line["wavelength"])
        line["intensity"] = float(line["intensity"])
        line["vacuum"] = True

    return lines


def nist_lines(
    element,
    states=["I", "II"],
    source=LineSource.NIST_STRONG,
    only_accurate=True,
) -> List[Dict]:
    """Load NIST reference lines for the specified element and ionisation states.

    Parameters
    ----------
    element : str
        Element to load from line list
    states : list, optional
        List of ionisation states, by default ["I", "II"]
    source : LineSource, optional
        Whether to load all or only strong lines, by default LineSource.NIST_STRONG
    only_accurate : bool, optional
        Only load lines with accuracies, by default True but ignored if NIST_STRONG

    Returns
    -------
    List[Dict]
        List of reference line dictionaries sorted by wavelength
    """
    files = nist_files(element, states, source)

    if (source == LineSource.NIST_STRONG) and only_accurate:
        only_accurate = False
        logger.debug(
            "Disabling accurate line filter as NIST strong lines do not have an associated accuracy."
        )

    lines = []

    for file in files:
        lines.extend(
            open_line_list(file, element=element, accurate=only_accurate)
        )

    lines.sort(key=lambda x: x["wavelength"])

    return lines
