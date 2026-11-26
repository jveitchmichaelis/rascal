import os
from functools import partialmethod

import pytest
import yaml

# Suppress tqdm output
from tqdm import tqdm

from rascal.atlas import Atlas, AtlasCollection

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

base_dir = os.path.dirname(__file__)


def test_load_atlas_from_yaml_file():
    yaml_config = os.path.join(base_dir, "test_config.yaml")

    with open(yaml_config, "r") as stream:
        config = yaml.safe_load(stream)["atlases"]

    atlases = [Atlas(**c) for c in config]
    assert len(atlases) == 2
    assert all(len(a) > 0 for a in atlases)


def test_atlas_collection_from_yaml_file():
    yaml_config = os.path.join(base_dir, "test_config.yaml")
    collection = AtlasCollection.from_config(yaml_config)

    assert len(collection.atlases) == 2
    assert len(collection.atlas_lines) > 0
    assert collection.wavelengths == sorted(collection.wavelengths)


def test_atlas_collection_from_dict():
    config = {
        "atlases": [
            {
                "element": "Test",
                "source": "manual",
                "wavelengths": [4000.0, 5000.0, 6000.0],
                "min_wavelength": 3500.0,
                "max_wavelength": 6500.0,
            },
            {
                "element": "Ar",
                "min_wavelength": 7000.0,
                "max_wavelength": 8000.0,
                "min_intensity": 50.0,
            },
        ]
    }
    collection = AtlasCollection.from_config(config)

    assert len(collection.atlases) == 2
    assert 4000.0 in collection.wavelengths
    assert 5000.0 in collection.wavelengths
    assert 6000.0 in collection.wavelengths


def test_atlas_collection_filtering():
    config = {
        "atlases": [
            {
                "element": "A",
                "source": "manual",
                "wavelengths": [4000.0, 5000.0, 6000.0],
                "min_wavelength": 3500.0,
                "max_wavelength": 6500.0,
            },
            {
                "element": "B",
                "source": "manual",
                "wavelengths": [4500.0, 5500.0],
                "min_wavelength": 3500.0,
                "max_wavelength": 6500.0,
            },
        ]
    }
    collection = AtlasCollection(
        atlases=config["atlases"],
        min_wavelength=4200.0,
        max_wavelength=5800.0,
        exclude_elements=["B"],
        exclude_wavelengths=[5000.0],
        exclude_tolerance=10.0,
    )
    assert collection.wavelengths == []


def test_atlas_collection_from_config_invalid_type():
    with pytest.raises(NotImplementedError):
        AtlasCollection.from_config(12345)


def test_atlas_unknown_source_raises():
    with pytest.raises(NotImplementedError):
        Atlas(
            element="He",
            source="unknown",
            min_wavelength=4000.0,
            max_wavelength=7000.0,
        )
