import os
from functools import partialmethod

import numpy as np
import pkg_resources
import pytest
import yaml

# Suppress tqdm output
from tqdm import tqdm

from rascal.atlas import Atlas

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

base_dir = os.path.dirname(__file__)


def test_load_atlas_from_yaml_file():
    atlas = Atlas()
    atlas.load_config(
        yaml_config=pkg_resources.resource_filename(
            "rascal", "../../atlas_yaml_template.yaml"
        )
    )


def test_load_atlas_from_pyyaml_object():
    with open(
        pkg_resources.resource_filename(
            "rascal", "../../atlas_yaml_template.yaml"
        ),
        "r",
    ) as stream:
        yaml_object = yaml.safe_load(stream)
        atlas = Atlas()
        atlas.load_config(yaml_config=yaml_object, y_type="object")


def test_load_atlas_config_user_linelist():
    with open(
        pkg_resources.resource_filename(
            "rascal", "../../atlas_yaml_template.yaml"
        ),
        "r",
    ) as stream:
        yaml_object = yaml.safe_load(stream)
        yaml_object["linelist"] = "user"
        yaml_object["element_list"] = np.array(["Xe"] * 10)
        yaml_object["wavelength_list"] = np.arange(10)
        atlas = Atlas()
        atlas.load_config(yaml_config=yaml_object, y_type="object")


@pytest.mark.xfail()
def test_load_atlas_config_expect_fail_ytype():
    atlas = Atlas()
    atlas.load_config(np.arange(100), y_type="bla")


@pytest.mark.xfail()
def test_load_atlas_config_expect_fail_linelist_type():
    with open(
        pkg_resources.resource_filename(
            "rascal", "../../atlas_yaml_template.yaml"
        ),
        "r",
    ) as stream:
        yaml_object = yaml.safe_load(stream)
        yaml_object["linelist"] = "blabla"
        atlas = Atlas()
        atlas.load_config(yaml_config=yaml_object, y_type="object")


def test_save_atlas_config():
    with open(
        pkg_resources.resource_filename(
            "rascal", "../../atlas_yaml_template.yaml"
        ),
        "r",
    ) as stream:
        yaml_object = yaml.safe_load(stream)
        atlas = Atlas()
        atlas.load_config(yaml_config=yaml_object, y_type="object")
        atlas.save_config(
            os.path.join(base_dir, "test_output", "test_atlas_config.yaml")
        )
