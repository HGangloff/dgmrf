import pytest
import os


def pytest_configure(config):
    os.environ["JAX_PLATFORMS"] = "cpu"
