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
    yaml_config = os.path.join(os.path.dirname(__file__), "test_config.yaml")

    with open(yaml_config, "r") as stream:
        config = yaml.safe_load(stream)["atlases"]

        for c in config:
            _ = Atlas(**c)


def test_load_atlas_from_pyyaml_object():
    pass


def test_load_atlas_config_user_linelist():
    pass


@pytest.mark.xfail()
def test_load_atlas_config_expect_fail_ytype():
    pass


@pytest.mark.xfail()
def test_load_atlas_config_expect_fail_linelist_type():
    pass


def test_save_atlas_config():
    pass
