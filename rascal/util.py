import numpy as np
import pkg_resources


def load_calibration_lines(elements,
                               min_wavelength=1000.,
                               max_wavelength=10000.):
        '''
        https://apps.dtic.mil/dtic/tr/fulltext/u2/a105494.pdf
        '''

        if isinstance(elements, str):
            elements = [elements]

        lines = []
        line_elements = []
        line_strengths = []

        for arc in elements:
            file_path = pkg_resources.resource_filename('rascal', 'arc_lines/{}.csv'.format(arc.lower()))

            with open(file_path, 'r') as f:

                f.readline()
                for l in f.readlines():
                    if l[0] == '#':
                        continue
                        
                    data = l.split(',')
                    if len(data) > 2:
                        line, strength, source = data[:3]
                        line_strengths.append(float(strength))
                    else:
                        line, source = data[:2]
                        line_strengths.append(0)
                    
                    lines.append(float(line))
                    line_elements.append(source)
       
        cal_lines = np.array(lines)
        cal_elements = np.array(line_elements)
        cal_strengths = np.array(line_strengths)

        # Get only lines within the requested wavelength
        mask = (cal_lines > min_wavelength) * (cal_lines < max_wavelength)
        return cal_lines[mask], cal_elements[mask], cal_strengths[mask]