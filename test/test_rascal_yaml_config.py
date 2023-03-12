import os
from functools import partialmethod

import numpy as np
import pkg_resources
import pytest
import yaml
from rascal import calibrator

# Suppress tqdm output
from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

base_dir = os.path.dirname(__file__)


def test_load_rascal_from_yaml_file():
    c = calibrator.Calibrator()
    c.load_config(
        yaml_config=pkg_resources.resource_filename(
            "rascal", "../../rascal_yaml_template.yaml"
        )
    )


def test_load_rascal_from_pyyaml_object():
    with open(
        pkg_resources.resource_filename(
            "rascal", "../../rascal_yaml_template.yaml"
        ),
        "r",
    ) as stream:
        yaml_object = yaml.safe_load(stream)
        c = calibrator.Calibrator()
        c.load_config(yaml_config=yaml_object, y_type="object")


@pytest.mark.xfail()
def test_load_rascal_config_expect_fail_ytype():
    c = calibrator.Calibrator()
    c.load_config(np.arange(100), y_type="bla")


def test_save_rascal_config():
    with open(
        pkg_resources.resource_filename(
            "rascal", "../../rascal_yaml_template.yaml"
        ),
        "r",
    ) as stream:
        yaml_object = yaml.safe_load(stream)
        c = calibrator.Calibrator()
        c.load_config(yaml_config=yaml_object, y_type="object")
        c.save_config(
            os.path.join(base_dir, "test_output", "test_rascal_config.yaml")
        )
