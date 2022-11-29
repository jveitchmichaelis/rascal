import numpy as np
from .util import load_calibration_lines
from .util import vacuum_to_air_wavelength


class AtlasLine:
    def __init__(self, wavelength, element=None, intensity=None, source=None):
        self.wavelength = wavelength
        self.element = element
        self.intensity = intensity
        self.source = source


class Atlas:
    def __init__(
        self,
        elements=None,
        min_atlas_wavelength=3000.0,
        max_atlas_wavelength=5000.0,
        range_tolerance=500,
        min_intensity=10.0,
        min_distance=10.0,
        vacuum=False,
        pressure=101325.0,
        temperature=273.15,
        relative_humidity=0.0,
        linelist="nist",
    ):
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

        Parameters
        ----------
        elements: string or list of strings
            Chemical symbol, case insensitive
        min_atlas_wavelength: float (default: None)
            Minimum wavelength of the arc lines.
        max_atlas_wavelength: float (default: None)
            Maximum wavelength of the arc lines.
        range_tolerance: float (default 500.)
            Range tolerance to add to min/max wavelengths
        min_intensity: float (default: None)
            Minimum intensity of the arc lines. Refer to NIST for the
            intensity.
        min_distance: float (default: None)
            Minimum separation between neighbouring arc lines.
        vacuum: boolean
            Set to True if the light path from the arc lamb to the detector
            plane is entirely in vacuum.
        pressure: float
            Pressure when the observation took place, in Pascal.
            If it is not known, we suggest you to assume 10% decrement per
            1000 meter altitude.
        temperature: float
            Temperature when the observation took place, in Kelvin.
        relative_humidity: float
            In percentage.

        """

        self.atlas_lines = []
        self.min_atlas_wavelength = min_atlas_wavelength
        self.max_atlas_wavelength = max_atlas_wavelength
        self.min_intensity = min_intensity
        self.min_distance = min_distance
        self.range_tolerance = range_tolerance

        if elements is not None:
            self.add(
                elements=elements,
                min_atlas_wavelength=min_atlas_wavelength,
                max_atlas_wavelength=max_atlas_wavelength,
                min_intensity=min_intensity,
                min_distance=min_distance,
                vacuum=vacuum,
                pressure=pressure,
                temperature=temperature,
                relative_humidity=relative_humidity,
                linelist=linelist,
            )

    def add(
        self,
        elements=None,
        min_atlas_wavelength=None,
        max_atlas_wavelength=None,
        min_intensity=10.0,
        min_distance=10.0,
        vacuum=False,
        pressure=101325.0,
        temperature=273.15,
        relative_humidity=0.0,
        linelist="nist",
        brightest_n_lines=1000,
    ):
        """
        Adds arc lines to the atlas

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

        Parameters
        ----------
        elements: string or list of strings
            Chemical symbol, case insensitive
        min_atlas_wavelength: float (default: None)
            Minimum wavelength of the arc lines.
        max_atlas_wavelength: float (default: None)
            Maximum wavelength of the arc lines.
        min_intensity: float (default: None)
            Minimum intensity of the arc lines. Refer to NIST for the
            intensity.
        min_distance: float (default: None)
            Minimum separation between neighbouring arc lines.
        vacuum: boolean
            Set to True if the light path from the arc lamb to the detector
            plane is entirely in vacuum.
        pressure: float
            Pressure when the observation took place, in Pascal.
            If it is not known, we suggest you to assume 10% decrement per
            1000 meter altitude.
        temperature: float
            Temperature when the observation took place, in Kelvin.
        relative_humidity: float
            In percentage.

        """

        if min_atlas_wavelength is None:

            min_atlas_wavelength = (
                self.min_atlas_wavelength - self.range_tolerance
            )

        if max_atlas_wavelength is None:

            max_atlas_wavelength = (
                self.max_atlas_wavelength + self.range_tolerance
            )

        if not np.isfinite(min_atlas_wavelength):

            raise ValueError(
                "min_atlas_wavelength has to be finite or None. "
                "{} is given.".format(min_atlas_wavelength)
            )

        if not np.isfinite(max_atlas_wavelength):

            raise ValueError(
                "max_atlas_wavelength has to be finite or None. "
                "{} is given.".format(max_atlas_wavelength)
            )

        if isinstance(elements, str):

            elements = [elements]

        if elements is not None:

            for element in elements:

                (
                    atlas_elements_tmp,
                    atlas_tmp,
                    atlas_intensities_tmp,
                ) = load_calibration_lines(
                    elements=element,
                    min_atlas_wavelength=min_atlas_wavelength,
                    max_atlas_wavelength=max_atlas_wavelength,
                    min_intensity=min_intensity,
                    min_distance=min_distance,
                    vacuum=vacuum,
                    pressure=pressure,
                    temperature=temperature,
                    relative_humidity=relative_humidity,
                    linelist=linelist,
                    brightest_n_lines=brightest_n_lines,
                )

                for element, line, intensity in list(
                    zip(atlas_elements_tmp, atlas_tmp, atlas_intensities_tmp)
                ):
                    self.atlas_lines.append(
                        AtlasLine(line, element, intensity, "NIST")
                    )

    def add_user_atlas(
        self,
        elements,
        wavelengths,
        intensities=None,
        vacuum=False,
        pressure=101325.0,
        temperature=273.15,
        relative_humidity=0.0,
    ):
        """
        Add a single or list of arc lines. Each arc line should have an
        element label associated with it. It is recommended that you use
        a standard periodic table abbreviation (e.g. 'Hg'), but it makes
        no difference to the fitting process.

        The vacuum to air wavelength conversion is deafult to False because
        observatories usually provide the line lists in the respective air
        wavelength, as the corrections from temperature and humidity are
        small. See https://emtoolbox.nist.gov/Wavelength/Documentation.asp

        Parameters
        ----------
        elements: list/str
            Elements (required). Preferably a standard (i.e. periodic table)
            name for convenience with built-in atlases
        wavelengths: list/float
            Wavelengths to add (Angstrom)
        intensities: list/float
            Relative line intensities (NIST value)
        vacuum: boolean
            Set to true to convert the input wavelength to air-wavelengths
            based on the given pressure, temperature and humidity.
        pressure: float
            Pressure when the observation took place, in Pascal.
            If it is not known, assume 10% decrement per 1000 meter altitude
        temperature: float
            Temperature when the observation took place, in Kelvin.
        relative_humidity: float
            In percentage.

        """

        if not isinstance(elements, list):

            elements = list(elements)

        if not isinstance(wavelengths, list):

            wavelengths = list(wavelengths)

        if intensities is None:

            intensities = [0] * len(wavelengths)

        else:

            if not isinstance(intensities, list):

                intensities = list(intensities)

        assert len(elements) == len(wavelengths), ValueError(
            "Input elements and wavelengths have different length."
        )
        assert len(elements) == len(intensities), ValueError(
            "Input elements and intensities have different length."
        )

        if vacuum:

            wavelengths = vacuum_to_air_wavelength(
                wavelengths, temperature, pressure, relative_humidity
            )

        self.min_atlas_wavelength = min(wavelengths)
        self.max_atlas_wavelength = max(wavelengths)

        for element, line, intensity in list(
            zip(elements, wavelengths, intensities)
        ):
            self.atlas_lines.append(AtlasLine(line, element, intensity))

    def get_lines(self):
        """
        Returns a list of line wavelengths in the atlas

        Returns
            wavelength_list: list

        """
        return [line.wavelength for line in self.atlas_lines]

    def get_elements(self):
        """
        Returns a list of per-line elements in the atlas

        Returns
            element_list: list

        """

        return [line.element for line in self.atlas_lines]

    def get_intensities(self):
        """
        Returns a list of per-line intensities in the atlas

        Returns
            intensity_list: list

        """

        return [line.intensity for line in self.atlas_lines]

    def remove_atlas_lines_range(self, wavelength, tolerance=10):
        """
        Remove arc lines within a certain wavelength range.

        Parameters
        ----------
        wavelength: float
            Wavelength to remove (Angstrom)
        tolerance: float
            Tolerance around this wavelength where atlas lines will be removed

        """

        for atlas_line in self.atlas_lines:

            if abs(atlas_line.wavelength - wavelength) < tolerance:

                self.atlas_lines.remove(atlas_line)

    def list(self):
        """
        List all the lines loaded to the Calibrator.

        """

        for line in self.atlas_lines:

            print(
                "Element "
                + str(line.element)
                + " at "
                + str(line.wavelength)
                + " with intensity "
                + str(line.intensity)
            )

    def clear(self):
        """
        Remove all the lines loaded to the Calibrator.

        """

        self.atlas_lines.clear()

    def __len__(self):
        return len(self.atlas_lines)
