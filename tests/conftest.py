import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure root path is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app

@pytest.fixture(scope="session")
def client():
    """Shared FastAPI test client fixture."""
    with TestClient(app) as test_client:
        yield test_client
