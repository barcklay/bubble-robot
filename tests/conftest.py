import pytest

from bubble_robot.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()
