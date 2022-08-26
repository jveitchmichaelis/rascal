import pkg_resources
import unittest.mock as mock


import pytest


@pytest.mark.xfail("ImportError")
@mock.patch("pkg_resources.get_distribution")
def test_should_fail_with_distribution_not_found(mock_require):

    mock_require.side_effect = pkg_resources.DistributionNotFound()
    import rascal
