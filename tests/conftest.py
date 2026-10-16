from typing import Iterator

import pytest

from src.core.job import RunJob


@pytest.fixture
def job() -> Iterator[RunJob]:
    """Provide a fresh, empty benchmark run job."""
    yield RunJob()
