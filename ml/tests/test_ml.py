"""
Unit and regression test for the ml package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import ml


def test_ml_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "ml" in sys.modules
