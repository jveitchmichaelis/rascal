import os
import re

import numpy as np

# The nist_raw.txt is obtained from
# https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=&limits_type=0&low_w=100&upp_w=30000&unit=0&de=0&format=1&line_out=3&remove_js=on&no_spaces=on&en_unit=0&output=0&page_size=15&show_obs_wl=1&order_out=0&max_low_enrg=&show_av=3&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&intens_out=on&max_str=&allowed_out=1&min_accur=&min_intens=&submit=Retrieve+Data
lines = []
with open("nist_raw.txt", "r") as raw_string:
    # skip the header
    [raw_string.readline() for i in range(7)]
    for line in raw_string.readlines():
        lines.append(np.array(line.split("|")))

data = np.vstack(lines)

elements = data[:, 0]
wavelengths = data[:, 1]
intensities = data[:, 2]

mask = elements == "          "

elements = elements[~mask]
wavelengths = wavelengths[~mask]
intensities = intensities[~mask]
states = np.empty(len(elements), dtype=object)

for i, (e, w, inten) in enumerate(zip(elements, wavelengths, intensities)):
    elements[i], states[i] = e.split(" ")[:2]
    inten_tmp = re.sub("[^a-zA-Z0-9]+", "", inten)
    try:
        float(inten_tmp)
    except:
        inten_tmp = 0.0
    intensities[i] = float(inten_tmp)
    wavelengths[i] = float(w)

states = states.astype("str")
elements = elements.astype("str")
intensities = intensities.astype("str")
wavelengths = wavelengths.astype("str")

table = "#element,wavelength,intensity,state\n"
for e, w, i, s in zip(elements, wavelengths, intensities, states):
    table += e + "," + w + "," + i + "," + s + "\n"

with open("nist_clean.csv", "w+") as f:
    f.write(table)
