"""
Tests for input sanitization utilities.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sanitize import sanitize_input, sanitize_id


class TestSanitizeInput:
    """Test text input sanitization."""

    def test_normal_text_unchanged(self):
        assert sanitize_input("Hello world") == "Hello world"

    def test_strips_control_characters(self):
        result = sanitize_input("Hello\x00\x01\x02world")
        assert "\x00" not in result
        assert result == "Helloworld"

    def test_preserves_newlines_and_tabs(self):
        result = sanitize_input("Line 1\nLine 2\tTabbed")
        assert "\n" in result
        assert "\t" in result

    def test_truncates_to_max_length(self):
        long_text = "x" * 5000
        result = sanitize_input(long_text, max_length=4000)
        assert len(result) == 4000

    def test_strips_whitespace(self):
        assert sanitize_input("  hello  ") == "hello"

    def test_empty_string(self):
        assert sanitize_input("") == ""

    def test_none_returns_empty(self):
        assert sanitize_input(None) == ""


class TestSanitizeId:
    """Test ID sanitization."""

    def test_normal_id_unchanged(self):
        assert sanitize_id("user-123_abc") == "user-123_abc"

    def test_strips_special_characters(self):
        result = sanitize_id("user@#$%123")
        assert result == "user123"

    def test_truncates_to_max_length(self):
        result = sanitize_id("x" * 200, max_length=100)
        assert len(result) == 100

    def test_empty_string(self):
        assert sanitize_id("") == ""
