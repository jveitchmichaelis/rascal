import importlib
import importlib.metadata
import unittest.mock as mock
from functools import partialmethod

import pytest

# Suppress tqdm output
from tqdm import tqdm

import rascal

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


@mock.patch("importlib.metadata.version")
def test_should_fail_with_distribution_not_found(mock_require):
    mock_require.side_effect = importlib.metadata.PackageNotFoundError(
        "rascal"
    )
    with pytest.raises(ImportError):
        importlib.reload(rascal)
