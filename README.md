# Rascal

Rascal is a library for automated spectrometer wavelength calibration. It has been designed primarily for astrophysics applications, but should be usable with spectra captured from any similar spectrometer.

Given a set of peaks located in your spectrum, Rascal will attempt to determine a model for your spectrometer to convert between pixels and wavelengths.

Unlike other calibration methods, rascal does not require you to manually select lines in your spectrum. Ideally you should know  approximate parameters about your system, namely:

* What arc lamp was used (e.g. Xe, Hg, Ar, CuNeAr)
* What the dispersion of your spectrometer is (i.e. angstroms/pixel)
* The spectral range of your system, and the starting wavelength

You don't need to know the dispersion and start wavelength exactly. Often this information is provided by the observatory, but if you don't know it, you can take a rough guess. The closer you are to the actual system settings, the more likely it is that Rascal will be able to solve the calibration. Blind calibration, where no parameters are known, is possible but challenging currently. If you don't know the lamp, you can try iterating over the various combinations of sources. Generally when you do get a correct fit, with most astronomical instruments the errors will be extremely low.

## Testing

To run the unit test suite without installing rascal, `cd` to the root directory and run:

```
python -m pytest test
```

To view logging output during testing, run:

```
python -m pytest test -s
```
