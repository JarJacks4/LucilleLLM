"""
LucilleLLM - Input Sanitization Utilities

Sanitizes user-facing text inputs before processing.
Strips control characters and enforces length limits.
"""

import re

# Control characters excluding tab, newline, carriage return
_CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def sanitize_input(text: str, max_length: int = 4000) -> str:
    """
    Sanitize user input text.

    - Truncates to max_length
    - Strips control characters (keeps tabs, newlines)
    - Strips leading/trailing whitespace
    """
    if not text:
        return ""
    text = text[:max_length]
    text = _CONTROL_CHARS.sub('', text)
    return text.strip()


def sanitize_id(value: str, max_length: int = 100) -> str:
    """
    Sanitize an ID string (user_id, session_id, etc.).

    - Only allows alphanumeric, hyphens, underscores
    - Truncates to max_length
    """
    if not value:
        return ""
    value = value[:max_length]
    value = re.sub(r'[^a-zA-Z0-9_\-]', '', value)
    return value
