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
    assert s.min_wavelength == 200.0
    assert s.max_wavelength == 1200.0


def test_initialisation():
    s = SyntheticSpectrum([1, 2, 3])
    assert s.degree == 2
    assert s.model is not None
    assert s.min_wavelength == 200.0
    assert s.max_wavelength == 1200.0


def test_constant_model():
    s = SyntheticSpectrum(np.arange(1))
    assert s.degree == 0

    s.get_pixels(wave1)


@pytest.mark.xfail()
def test_linear_model():
    s = SyntheticSpectrum(np.arange(2))
    assert s.degree == 2

    s.get_pixels(wave1)


def test_cubic_model():
    s = SyntheticSpectrum(np.arange(4))
    assert s.degree == 3

    p, _ = s.get_pixels(wave1)
    assert len(p) == np.sum((wave1 > 200.0) & (wave1 < 1200.0))


def test_switching_model():
    s = SyntheticSpectrum(np.arange(3))
    assert s.degree == 2

    p, _ = s.get_pixels(wave1)
    assert len(p) == np.sum((wave1 > 200.0) & (wave1 < 1200.0))

    s.set_model(np.arange(5))
    assert s.degree == 4

    p, _ = s.get_pixels(wave1)
    assert len(p) == np.sum((wave1 > 200.0) & (wave1 < 1200.0))


def test_switching_wavelength_ranges():
    s = SyntheticSpectrum(np.arange(3))
    assert s.degree == 2

    p, _ = s.get_pixels(wave1)
    assert len(p) == np.sum((wave1 > 200.0) & (wave1 < 1200.0))

    s.set_wavelength_limit(min_wavelength=500.0)
    p, _ = s.get_pixels(wave1)
    assert len(p) == np.sum((wave1 > 500.0) & (wave1 < 1200.0))

    s.set_wavelength_limit(max_wavelength=1000.0)
    p, _ = s.get_pixels(wave1)
    assert len(p) == np.sum((wave1 > 500.0) & (wave1 < 1000.0))


@pytest.mark.xfail()
def test_wrong_coefficients():
    SyntheticSpectrum(1.0)


@pytest.mark.xfail()
def test_set_wrong_wavelength_limit():
    s = SyntheticSpectrum(np.arange(3))
    assert s.degree == 2

    s.set_wavelength_limit(min_wavelength=500.0, max_wavelength=300.0)


@pytest.mark.xfail()
def test_set_wrong_min_wavelength_limit():
    s = SyntheticSpectrum(np.arange(3))
    assert s.degree == 2

    s.set_wavelength_limit(min_wavelength="Hellow")


@pytest.mark.xfail()
def test_set_wrong_max_wavelength_limit():
    s = SyntheticSpectrum(np.arange(3))
    assert s.degree == 2

    s.set_wavelength_limit(max_wavelength="World")


@pytest.mark.xfail()
def test_wrong_wavelengths_to_be_inverted():
    s = SyntheticSpectrum(np.arange(3))
    assert s.degree == 2

    s.get_pixels(1.0)
