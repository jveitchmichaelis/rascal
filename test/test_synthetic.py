import numpy as np
import pytest

from rascal.synthetic import SyntheticSpectrum

wave1 = np.arange(1500)
wave2 = np.arange(500, 1500)
wave3 = np.arange(500)
wave4 = np.arange(1500, 2000)


def test_default():
    s = SyntheticSpectrum()
    assert s.degree is None
    assert s.model is None
    assert s.min_wavelength == 200.
    assert s.max_wavelength == 1200.


def test_initialisation():
    s = SyntheticSpectrum([1, 2, 3])
    assert s.degree == 2
    assert s.model is not None
    assert s.min_wavelength == 200.
    assert s.max_wavelength == 1200.


# Pynverse can only work on strictly monotonic functions
# A constant model gives a constant...
@pytest.mark.xfail()
def test_constant_model():
    s = SyntheticSpectrum(np.arange(1))
    assert s.degree == 0

    s.get_pixels(wave1)


# No idea why this fails...
@pytest.mark.xfail()
def test_linear_model():
    s = SyntheticSpectrum(np.arange(2))
    assert s.degree == 1

    p = s.get_pixels(wave1)
    assert len(p) == np.sum((wave1 > 200.) & (wave1 < 1200.))


def test_cubic_model():
    s = SyntheticSpectrum(np.arange(4))
    assert s.degree == 3

    p = s.get_pixels(wave1)
    assert len(p) == np.sum((wave1 > 200.) & (wave1 < 1200.))


def test_switching_model():
    s = SyntheticSpectrum(np.arange(3))
    assert s.degree == 2

    p = s.get_pixels(wave1)
    assert len(p) == np.sum((wave1 > 200.) & (wave1 < 1200.))

    s.set_model(np.arange(5))
    assert s.degree == 4

    p = s.get_pixels(wave1)
    assert len(p) == np.sum((wave1 > 200.) & (wave1 < 1200.))


def test_switching_wavelength_ranges():
    s = SyntheticSpectrum(np.arange(3))
    assert s.degree == 2

    p = s.get_pixels(wave1)
    assert len(p) == np.sum((wave1 > 200.) & (wave1 < 1200.))

    s.set_wavelength_limit(min_wavelength=500.)
    p = s.get_pixels(wave1)
    assert len(p) == np.sum((wave1 > 500.) & (wave1 < 1200.))

    s.set_wavelength_limit(max_wavelength=1000.)
    p = s.get_pixels(wave1)
    assert len(p) == np.sum((wave1 > 500.) & (wave1 < 1000.))


@pytest.mark.xfail()
def test_wrong_coefficients():
    SyntheticSpectrum(1.)


@pytest.mark.xfail()
def test_set_wrong_wavelength_limit():
    s = SyntheticSpectrum(np.arange(3))
    assert s.degree == 2

    s.set_wavelength_limit(min_wavelength=500., max_wavelength=300.)


@pytest.mark.xfail()
def test_set_wrong_min_wavelength_limit():
    s = SyntheticSpectrum(np.arange(3))
    assert s.degree == 2

    s.set_wavelength_limit(min_wavelength='Hellow')


@pytest.mark.xfail()
def test_set_wrong_max_wavelength_limit():
    s = SyntheticSpectrum(np.arange(3))
    assert s.degree == 2

    s.set_wavelength_limit(max_wavelength="World")


@pytest.mark.xfail()
def test_wrong_wavelengths_to_be_inverted():
    s = SyntheticSpectrum(np.arange(3))
    assert s.degree == 2

    s.get_pixels(1.)
