import unittest.mock as mock
from functools import partialmethod

import pkg_resources
import pytest

# Suppress tqdm output
from tqdm import tqdm

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


@pytest.mark.xfail("ImportError")
@mock.patch("pkg_resources.get_distribution")
def test_should_fail_with_distribution_not_found(mock_require):

    mock_require.side_effect = pkg_resources.DistributionNotFound()
    import rascal
