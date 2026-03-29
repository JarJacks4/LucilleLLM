"""
Tests for configuration system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config


class TestConfigDefaults:
    """Verify all config defaults are set correctly."""

    def setup_method(self):
        self.config = get_config()

    def test_default_model(self):
        assert self.config.OPENAI_MODEL == "gpt-4o-mini"

    def test_default_rate_limits(self):
        assert self.config.RATE_LIMIT_CHAT == 10
        assert self.config.RATE_LIMIT_GLOBAL == 100

    def test_assessment_defaults(self):
        assert self.config.ASSESSMENT_ENABLED is True
        assert self.config.ASSESSMENT_REMINDER_DAYS == 14
        assert self.config.ASSESSMENT_PHQ9_CONCERN_THRESHOLD == 10
        assert self.config.ASSESSMENT_GAD7_CONCERN_THRESHOLD == 10
        assert self.config.ASSESSMENT_WHO5_CONCERN_THRESHOLD == 50

    def test_safety_defaults(self):
        assert self.config.ESCALATION_ENABLED is True
        assert self.config.AUDIT_LOG_ENABLED is True

    def test_retention_defaults(self):
        assert self.config.RETENTION_AUDIT_LOG_DAYS == 2555  # 7 years
        assert self.config.RETENTION_CHAT_SESSIONS_DAYS == 365

    def test_config_is_frozen(self):
        """Config should be immutable (frozen dataclass)."""
        import dataclasses
        assert dataclasses.is_dataclass(self.config)
