"""
LucilleLLM - Retry Utilities

Provides retry decorators for external service calls (OpenAI, Firebase, GCS).
Uses tenacity for exponential backoff with jitter.
"""

import logging

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


def openai_retry(func):
    """
    Retry decorator for OpenAI API calls.
    Retries on rate limits, timeouts, and connection errors.
    3 attempts with exponential backoff (1s, 2s, 4s).
    """
    try:
        from openai import RateLimitError, APITimeoutError, APIConnectionError

        return retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(
                (RateLimitError, APITimeoutError, APIConnectionError)
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )(func)
    except ImportError:
        return func


def firebase_retry(func):
    """
    Retry decorator for Firebase/Firestore calls.
    Retries on transient Google Cloud errors.
    3 attempts with exponential backoff.
    """
    try:
        from google.api_core.exceptions import (
            ServiceUnavailable,
            DeadlineExceeded,
            InternalServerError,
        )

        return retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(
                (ServiceUnavailable, DeadlineExceeded, InternalServerError)
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )(func)
    except ImportError:
        return func
