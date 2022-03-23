# Rascal
[![Python package](https://github.com/jveitchmichaelis/rascal/actions/workflows/python-package.yml/badge.svg)](https://github.com/jveitchmichaelis/rascal/actions/workflows/python-package.yml)
[![Coverage Status](https://coveralls.io/repos/github/jveitchmichaelis/rascal/badge.svg?branch=main)](https://coveralls.io/github/jveitchmichaelis/rascal?branch=main)
[![Readthedocs Status](https://readthedocs.org/projects/rascal/badge/?version=latest&style=flat)](https://rascal.readthedocs.io/en/latest/)
[![PyPI version](https://badge.fury.io/py/rascal.svg)](https://badge.fury.io/py/rascal)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4117516.svg)](https://doi.org/10.5281/zenodo.4117516)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Rascal is a library for automated spectrometer wavelength calibration. It has been designed primarily for astrophysics applications, but should be usable with spectra captured from any similar spectrometer.

Given a set of peaks located in your spectrum, Rascal will attempt to determine a model for your spectrometer to convert between pixels and wavelengths.

Unlike other calibration methods, rascal does not require you to manually select lines in your spectrum. Ideally you should know  approximate parameters about your system, namely:

* What arc lamp was used (e.g. Xe, Hg, Ar, CuNeAr)
* What the dispersion of your spectrometer is (i.e. angstroms/pixel)
* The spectral range of your system, and the starting wavelength

You don't need to know the dispersion and start wavelength exactly. Often this information is provided by the observatory, but if you don't know it, you can take a rough guess. The closer you are to the actual system settings, the more likely it is that Rascal will be able to solve the calibration. Blind calibration, where no parameters are known, is possible but challenging currently. If you don't know the lamp, you can try iterating over the various combinations of sources. Generally when you do get a correct fit, with most astronomical instruments the errors will be extremely low.

More background information can be referred to this [arXiv article](https://ui.adsabs.harvard.edu/abs/2019arXiv191205883V/abstract).


## Dependencies
* python >= 3.6
* numpy
* scipy
* [astropy](https://github.com/astropy/astropy)
* [plotly](https://github.com/plotly/plotly.py) >= 4.0

## Installation
Instructions can be found [here](https://rascal.readthedocs.io/en/latest/installation/installation.html).

## Reporting issues/feature requests
Please use the [issue tracker](https://github.com/jveitchmichaelis/rascal/issues) to report any issues or support questions.

## Getting started
The [quickstart guide](https://rascal.readthedocs.io/en/latest/tutorial/quickstart.html) will show you how to reduce the example dataset.

## Contributing Code/Documentation
If you are interested in contributing code to the project, thank you! For those unfamiliar with the process of contributing to an open-source project, you may want to read through Github’s own short informational section on how to submit a [contribution](https://opensource.guide/how-to-contribute/#how-to-submit-a-contribution) or send me a message.

Style -- we now use black for formatting, you can easily set this up using a pre-commit hook.

```
pip install pre-commit
pre-commit install
```

## Disclaimer
We duplicate some of the relevant metadata, but we do not process the raw metadata. Some of the metadata this software creates contain full path to the files in your system, which most likely includes a user name on your machine. Please be advised it is your responsibility to be compliant with the privacy law(s) that you are oblidged to follow, and it is your responsibility to remove any metadata that may reveal personal information and/or provide information that can reveal any computing vulunerability.
