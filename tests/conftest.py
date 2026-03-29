"""
Shared test fixtures for LucilleLLM.

Provides mocked Firebase and OpenAI services so tests can run
without external dependencies.
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def mock_firebase():
    """Mock Firebase for all tests — no real Firestore needed."""
    mock_service = MagicMock()
    mock_service.db = None  # Simulate no Firebase connection
    with patch("firebase_service.get_firebase_service", return_value=mock_service):
        yield mock_service


@pytest.fixture
def mock_firebase_with_db():
    """Mock Firebase WITH a working db for tests that need Firestore operations."""
    mock_service = MagicMock()
    mock_db = MagicMock()
    mock_service.db = mock_db
    with patch("firebase_service.get_firebase_service", return_value=mock_service):
        yield mock_service, mock_db


@pytest.fixture
def config():
    """Get the app config (uses defaults, no env vars needed)."""
    from config import get_config
    return get_config()
