import numpy as np
from .util import load_calibration_lines
from .util import vacuum_to_air_wavelength

class Atlas:
    def __init__(self, elements=None,
                  min_atlas_wavelength=3000,
                  max_atlas_wavelength=5000,
                  range_tolerance=500,
                  min_intensity=10.,
                  min_distance=10.,
                  vacuum=False,
                  pressure=101325.,
                  temperature=273.15,
                  relative_humidity=0.):
                  
        '''
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

        '''

        self.elements = []
        self.lines = []
        self.intensities = []
        self.min_atlas_wavelength = min_atlas_wavelength
        self.max_atlas_wavelength = max_atlas_wavelength
        self.min_intensity=min_intensity
        self.min_distance=min_distance
        self.range_tolerance = range_tolerance

        if elements is not None:
            self.add(elements=elements,
                min_atlas_wavelength=min_atlas_wavelength,
                max_atlas_wavelength=max_atlas_wavelength,
                min_intensity=min_intensity,
                min_distance=min_distance,
                vacuum=vacuum,
                pressure=pressure,
                temperature=temperature,
                relative_humidity=relative_humidity)

    def add(self,elements=None,
                min_atlas_wavelength=None,
                max_atlas_wavelength=None,
                min_intensity=10.,
                min_distance=10.,
                vacuum=False,
                pressure=101325.,
                temperature=273.15,
                relative_humidity=0.):
        '''
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

        '''

        if min_atlas_wavelength is None:

            min_atlas_wavelength = self.min_atlas_wavelength - self.range_tolerance

        if max_atlas_wavelength is None:

            max_atlas_wavelength = self.max_atlas_wavelength + self.range_tolerance

        if isinstance(elements, str):

            elements = [elements]
        elif elements is not None:
            for element in elements:

                atlas_elements_tmp, atlas_tmp, atlas_intensities_tmp =\
                    load_calibration_lines(
                        element, min_atlas_wavelength, max_atlas_wavelength,
                        min_intensity, min_distance, vacuum, pressure, temperature,
                        relative_humidity)

                self.elements.extend(atlas_elements_tmp)
                self.lines.extend(atlas_tmp)
                self.intensities.extend(atlas_intensities_tmp)

    def add_user_atlas(self,
                       elements,
                       wavelengths,
                       intensities=None,
                       vacuum=False,
                       pressure=101325.,
                       temperature=273.15,
                       relative_humidity=0.):
        '''
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

        '''

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
            'Input elements and wavelengths have different length.')
        assert len(elements) == len(intensities), ValueError(
            'Input elements and intensities have different length.')

        if vacuum:

            wavelengths = vacuum_to_air_wavelength(wavelengths, temperature,
                                                   pressure, relative_humidity)

        self.elements.extend(elements)
        self.lines.extend(wavelengths)
        self.intensities.extend(intensities)

    def remove_atlas_lines_range(self, wavelength, tolerance=10):
        '''
        Remove arc lines within a certain wavelength range.

        Parameters
        ----------
        wavelength: float
            Wavelength to remove (Angstrom)
        tolerance: float
            Tolerance around this wavelength where atlas lines will be removed

        '''

        for i, line in enumerate(self.lines):

            if abs(line - wavelength) < tolerance:

                removed_element = self.elements.pop(i)
                removed_peak = self.lines.pop(i)
                self.intensities.pop(i)

                self.logger.info('Removed {} line: {} A'.format(
                    removed_element, removed_peak))

    def list(self):
        '''
        List all the lines loaded to the Calibrator.

        '''

        for i in range(len(self.lines)):

            print('Element ' + str(self.elements[i]) + ' at ' +
                  str(self.lines[i]) + ' with intensity ' +
                  str(self.intensities[i]))

    def clear(self):
        '''
        Remove all the lines loaded to the Calibrator.

        '''

        self.elements = []
        self.lines = []
        self.intensities = []